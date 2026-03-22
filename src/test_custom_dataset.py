import os
import sys
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from tqdm import tqdm

# =====================================================================
# 1. 导入 learning3d
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 如有需要手动加路径:
# sys.path.append(os.path.join(BASE_DIR, "../../learning3d_path"))

try:
    from learning3d.models import DGCNN, DCP, PointNet, PointNetLK, iPCRNet, PPFNet, RPMNet, PRNet
except ImportError as e:
    print("⚠️ 无法导入 learning3d 库，请检查环境变量或路径。")
    raise e

from registration import compute_registration_error
from surgical_registration_dataset import SurgicalRegistrationData


# =====================================================================
# 2. 数学工具
# =====================================================================
def get_transform_from_corres(P, Q):
    """根据对应点集计算 SVD 刚体变换, P->Q"""
    P_xyz = P[:, :3]
    Q_xyz = Q[:, :3]
    centroid_P = np.mean(P_xyz, axis=0)
    centroid_Q = np.mean(Q_xyz, axis=0)

    p = P_xyz - centroid_P
    q = Q_xyz - centroid_Q

    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_Q - R @ centroid_P
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def denormalize_transform(T_norm, centroid, scale):
    """把归一化空间变换还原到真实坐标"""
    R_norm = T_norm[:3, :3]
    t_norm = T_norm[:3, 3]

    R_real = R_norm
    t_real = scale * t_norm + centroid - R_norm @ centroid

    T_real = np.eye(4, dtype=np.float64)
    T_real[:3, :3] = R_real
    T_real[:3, 3] = t_real
    return T_real


def preprocess_dataset_sample(pts_t, pts_s, num_points=1024, normal_radius=0.1, normal_max_nn=30):
    """
    输出:
      t3,s3: [1,N,3] 归一化xyz
      t6,s6: [1,N,6] 归一化xyz + normals
      centroid_t, scale: 反归一化参数
    """
    pcd_t = o3d.geometry.PointCloud()
    pcd_s = o3d.geometry.PointCloud()
    pcd_t.points = o3d.utility.Vector3dVector(pts_t)
    pcd_s.points = o3d.utility.Vector3dVector(pts_s)

    pcd_t.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    pcd_s.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )

    pts_t_arr = np.asarray(pcd_t.points)
    nrm_t_arr = np.asarray(pcd_t.normals)
    pts_s_arr = np.asarray(pcd_s.points)
    nrm_s_arr = np.asarray(pcd_s.normals)

    def sample_data(pts, nrms, n):
        idx = np.random.choice(len(pts), n, replace=(len(pts) < n))
        return pts[idx], nrms[idx]

    pts_t_arr, nrm_t_arr = sample_data(pts_t_arr, nrm_t_arr, num_points)
    pts_s_arr, nrm_s_arr = sample_data(pts_s_arr, nrm_s_arr, num_points)

    # 用 target 归一化，保持相对位姿
    centroid_t = np.mean(pts_t_arr, axis=0)
    pts_t_norm = pts_t_arr - centroid_t
    pts_s_norm = pts_s_arr - centroid_t

    scale = np.max(np.sqrt(np.sum(pts_t_norm ** 2, axis=1)))
    scale = scale if scale > 0 else 1.0
    pts_t_norm = pts_t_norm / scale
    pts_s_norm = pts_s_norm / scale

    xyzn_t = np.concatenate([pts_t_norm, nrm_t_arr], axis=1)
    xyzn_s = np.concatenate([pts_s_norm, nrm_s_arr], axis=1)

    tensor_t3 = torch.tensor(pts_t_norm, dtype=torch.float32).unsqueeze(0)
    tensor_s3 = torch.tensor(pts_s_norm, dtype=torch.float32).unsqueeze(0)
    tensor_t6 = torch.tensor(xyzn_t, dtype=torch.float32).unsqueeze(0)
    tensor_s6 = torch.tensor(xyzn_s, dtype=torch.float32).unsqueeze(0)

    return tensor_t3, tensor_s3, tensor_t6, tensor_s6, centroid_t, scale


def refine_registration_icp(source_pts, target_pts, initial_transform, distance_threshold=0.05, max_iter=200):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pts)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)

    reg = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, distance_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return reg.transformation


# =====================================================================
# 3. 你训练时用到的 SurgicalRPMNet（测试必须一致）
# =====================================================================
def compute_source_tangent(src_xyz):
    # src_xyz: [B,N,3], 依赖轨迹顺序
    B, N, _ = src_xyz.shape
    if N < 3:
        return F.normalize(torch.zeros_like(src_xyz) + 1e-6, dim=-1)
    prev = src_xyz[:, :-2, :]
    nxt = src_xyz[:, 2:, :]
    tan_mid = F.normalize(nxt - prev, dim=-1)

    tan = torch.zeros_like(src_xyz)
    tan[:, 1:-1, :] = tan_mid
    tan[:, 0, :] = tan[:, 1, :]
    tan[:, -1, :] = tan[:, -2, :]
    return F.normalize(tan, dim=-1)


class TrajectoryFeatureAdapter(nn.Module):
    def __init__(self, feat_dim=96):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )

    def forward(self, src_xyz, src_feat):
        tan = compute_source_tangent(src_xyz)
        gate = self.mlp(tan)
        return src_feat * (1.0 + gate)


class SurgicalRPMNet(nn.Module):
    """
    兼容版本：
    - 如果底层 RPMNet 没有中间特征，仍可正常返回 R/t。
    """
    def __init__(self, rpmnet, feat_dim=96):
        super().__init__()
        self.rpmnet = rpmnet
        self.traj_adapter = TrajectoryFeatureAdapter(feat_dim=feat_dim)

    def _parse_output(self, raw_out):
        out = {}

        if isinstance(raw_out, dict):
            # 常见格式
            if 'est_R' in raw_out and 'est_t' in raw_out:
                out['R'] = raw_out['est_R']
                out['t'] = raw_out['est_t']
            elif 'R' in raw_out and 't' in raw_out:
                out['R'] = raw_out['R']
                out['t'] = raw_out['t']
            elif 'transformation' in raw_out:
                T = raw_out['transformation']
                out['R'] = T[:, :3, :3]
                out['t'] = T[:, :3, 3]
            elif 'transformed_source' in raw_out:
                # 用 source 与 transformed_source 回推
                out['transformed_source'] = raw_out['transformed_source']
            else:
                raise RuntimeError(f"Unknown dict output keys: {list(raw_out.keys())}")

            # 透传可能存在的中间量
            if 'src_feat' in raw_out:
                out['src_feat'] = raw_out['src_feat']
            if 'match_logits' in raw_out:
                out['match_logits'] = raw_out['match_logits']
            if 'corr_prob' in raw_out:
                out['corr_prob'] = raw_out['corr_prob']

        elif isinstance(raw_out, (tuple, list)):
            # 可能是 (R,t) 或 (T,)
            if len(raw_out) >= 2 and torch.is_tensor(raw_out[0]) and torch.is_tensor(raw_out[1]):
                if raw_out[0].dim() == 3 and raw_out[0].shape[-2:] == (3, 3):
                    out['R'], out['t'] = raw_out[0], raw_out[1]
                else:
                    raise RuntimeError("Tuple output not recognized.")
            elif len(raw_out) >= 1 and torch.is_tensor(raw_out[0]) and raw_out[0].shape[-2:] == (4, 4):
                T = raw_out[0]
                out['R'] = T[:, :3, :3]
                out['t'] = T[:, :3, 3]
            else:
                raise RuntimeError("List/Tuple output not recognized.")
        elif torch.is_tensor(raw_out):
            # 可能直接返回 [B,4,4]
            if raw_out.dim() == 3 and raw_out.shape[-2:] == (4, 4):
                out['R'] = raw_out[:, :3, :3]
                out['t'] = raw_out[:, :3, 3]
            else:
                raise RuntimeError(f"Tensor output not recognized: {raw_out.shape}")
        else:
            raise RuntimeError(f"Unsupported output type: {type(raw_out)}")

        return out

    def forward(self, template, source):
        # 注意：不传 return_intermediate，避免你之前报错
        raw_out = self.rpmnet(template, source)
        return self._parse_output(raw_out)


# =====================================================================
# 4. 模型加载与推理
# =====================================================================
def build_model(name):
    if name == 'DCP':
        return DCP(feature_model=DGCNN(emb_dims=512), cycle=True)
    elif name == 'PointNetLK':
        return PointNetLK(feature_model=PointNet(emb_dims=1024, use_bn=True))
    elif name == 'PCRNet':
        return iPCRNet(feature_model=PointNet(emb_dims=1024))
    elif name == 'RPMNet':
        return RPMNet(feature_model=PPFNet())
    elif name == 'SurgicalRPMNet':
        base = RPMNet(feature_model=PPFNet())
        return SurgicalRPMNet(base, feat_dim=96)
    elif name == 'PRNet':
        return PRNet(emb_dims=512, num_iters=3)
    else:
        return None


def extract_state_dict(checkpoint):
    """
    兼容:
    - {'state_dict': ...}
    - {'model': ...}
    - 纯 state_dict
    """
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        if 'model' in checkpoint:
            return checkpoint['model']
    return checkpoint


def load_model(name, path, device, strict=False):
    if not os.path.exists(path):
        print(f"❌ 权重不存在: {path}")
        return None

    model = build_model(name)
    if model is None:
        print(f"❌ 不支持的模型名: {name}")
        return None

    checkpoint = torch.load(path, map_location='cpu')
    state_dict = extract_state_dict(checkpoint)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if len(missing) > 0:
        print(f"[{name}] Missing keys: {len(missing)}")
        # print(missing)
    if len(unexpected) > 0:
        print(f"[{name}] Unexpected keys: {len(unexpected)}")
        # print(unexpected)

    return model.to(device).eval()


def run_model_inference(model, model_name, t_in, s_in, device):
    """
    返回归一化空间下 T_norm (4x4), 从 source -> template
    """
    source_np = s_in.cpu().numpy()[0, :, :3]

    with torch.no_grad():
        if model_name == 'PRNet':
            dummy_R = torch.eye(3).unsqueeze(0).to(device)
            dummy_t = torch.zeros(1, 3).to(device)
            output = model(t_in, s_in, dummy_R, dummy_t)
            R_ab = output['est_R'][0].cpu().numpy()
            t_ba = output['est_t'][0].cpu().numpy()
            t_ab = -np.dot(R_ab, t_ba)
            T_norm = np.eye(4, dtype=np.float64)
            T_norm[:3, :3] = R_ab
            T_norm[:3, 3] = t_ab
            return T_norm

        output = model(t_in, s_in)

        # dict
        if isinstance(output, dict):
            if 'R' in output and 't' in output:
                R = output['R'][0].detach().cpu().numpy()
                t = output['t'][0].detach().cpu().numpy()
                T_norm = np.eye(4, dtype=np.float64)
                T_norm[:3, :3] = R
                T_norm[:3, 3] = t
                return T_norm

            if 'est_R' in output and 'est_t' in output:
                R = output['est_R'][0].detach().cpu().numpy()
                t = output['est_t'][0].detach().cpu().numpy()
                T_norm = np.eye(4, dtype=np.float64)
                T_norm[:3, :3] = R
                T_norm[:3, 3] = t
                return T_norm

            if 'transformation' in output:
                T = output['transformation'][0].detach().cpu().numpy().astype(np.float64)
                return T

            if 'transformed_source' in output:
                dl_aligned_np = output['transformed_source'][0].detach().cpu().numpy()[:, :3]
                return get_transform_from_corres(source_np, dl_aligned_np)

        # tuple/list
        if isinstance(output, (tuple, list)):
            if len(output) >= 2 and torch.is_tensor(output[0]) and torch.is_tensor(output[1]):
                if output[0].dim() == 3 and output[0].shape[-2:] == (3, 3):
                    R = output[0][0].detach().cpu().numpy()
                    t = output[1][0].detach().cpu().numpy()
                    T_norm = np.eye(4, dtype=np.float64)
                    T_norm[:3, :3] = R
                    T_norm[:3, 3] = t
                    return T_norm
            if len(output) >= 1 and torch.is_tensor(output[0]) and output[0].shape[-2:] == (4, 4):
                return output[0][0].detach().cpu().numpy().astype(np.float64)

        # tensor [B,4,4]
        if torch.is_tensor(output) and output.dim() == 3 and output.shape[-2:] == (4, 4):
            return output[0].detach().cpu().numpy().astype(np.float64)

    raise RuntimeError("无法解析模型输出格式，请打印 output 的类型与 keys。")


def calculate_metrics(errors_list):
    errors = np.array(errors_list)
    rre_arr, rte_arr = errors[:, 0], errors[:, 1]

    rre_mse = np.mean(rre_arr ** 2)
    rre_mae = np.mean(np.abs(rre_arr))
    rte_mse = np.mean(rte_arr ** 2)
    rte_mae = np.mean(np.abs(rte_arr))

    # 成功标准（可调）
    success_mask = (rte_arr < 0.01) & (rre_arr < 1.0)
    success_rate = np.mean(success_mask) * 100.0

    return rre_mse, rre_mae, rte_mse, rte_mae, success_rate


# =====================================================================
# 5. 主函数
# =====================================================================
def main():
    parser = argparse.ArgumentParser()

    # 你可保留其他模型，这里重点是 RPMNet / SurgicalRPMNet
    parser.add_argument('--w_rpmnet', type=str, default='./learning3d/pretrained/exp_rpmnet/models/partial-trained.pth')
    parser.add_argument('--w_srpmnet', type=str, default='./checkpoints_surgical_rpmnet/best.pth',
                        help='新训练 SurgicalRPMNet 权重路径')

    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--icp_threshold', type=float, default=0.05)
    parser.add_argument('--icp_max_iter', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda:0')

    # 数据参数（可直接测 45/90/180）
    parser.add_argument('--template_points', type=int, default=2048)
    parser.add_argument('--source_points', type=int, default=2000)
    parser.add_argument('--angle_range', type=float, default=45.0)
    parser.add_argument('--translation_range', type=float, default=1.0)
    parser.add_argument('--noise_sigma', type=float, default=0.005)
    parser.add_argument('--coverage_ratio', type=float, default=0.90)

    # 预处理参数
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--normal_radius', type=float, default=0.1)
    parser.add_argument('--normal_max_nn', type=int, default=30)

    # 只测某个模型
    parser.add_argument('--only_model', type=str, default='',
                        choices=['', 'RPMNet', 'SurgicalRPMNet'])

    # state_dict 加载 strict
    parser.add_argument('--strict_load', action='store_true')

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("Loading dataset...")
    dataset = SurgicalRegistrationData(
        data_root="./learning3d/data//modelnet40_ply_hdf5_2048",
        partition='test',
        template_points=args.template_points,
        source_points=args.source_points,
        angle_range=args.angle_range,
        translation_range=args.translation_range,
        noise_sigma=args.noise_sigma,
        coverage_ratio=args.coverage_ratio,
    )

    num_samples = min(args.num_samples, len(dataset))

    tasks = []
    if args.only_model == 'RPMNet':
        tasks = [('RPMNet', args.w_rpmnet)]
    elif args.only_model == 'SurgicalRPMNet':
        tasks = [('SurgicalRPMNet', args.w_srpmnet)]
    else:
        tasks = [
            ('RPMNet', args.w_rpmnet),
            ('SurgicalRPMNet', args.w_srpmnet),
        ]

    valid_tasks = []
    for name, path in tasks:
        if os.path.exists(path):
            valid_tasks.append((name, path))
        else:
            print(f"⚠️ 跳过 {name}, 权重不存在: {path}")

    if not valid_tasks:
        print("❌ 没有可用权重，退出。")
        return

    results_summary = {}

    for model_name, weight_path in valid_tasks:
        print(f"\n⚙️ 正在评测 {model_name} ...")
        model = load_model(model_name, weight_path, device, strict=args.strict_load)
        if model is None:
            continue

        geo_errors = []
        icp_errors = []

        for i in tqdm(range(num_samples), desc=f"Testing {model_name}"):
            template_t, source_t, igt_t = dataset[i]

            # 真实坐标
            pts_t_real = template_t.numpy()
            pts_s_real = source_t.numpy()

            # 你的数据集里 igt 是 source_raw -> source_transformed
            # 若要 source_transformed -> template/raw，需要取逆
            T_gt = np.linalg.inv(igt_t.numpy())

            # 归一化输入
            t3, s3, t6, s6, centroid_t, scale = preprocess_dataset_sample(
                pts_t_real, pts_s_real,
                num_points=args.num_points,
                normal_radius=args.normal_radius,
                normal_max_nn=args.normal_max_nn
            )

            # RPMNet/SurgicalRPMNet 用 xyz+normal
            if model_name in ['RPMNet', 'SurgicalRPMNet']:
                t_in = t6.to(device)
                s_in = s6.to(device)
            else:
                t_in = t3.to(device)
                s_in = s3.to(device)

            # DL inference
            T_DL_norm = run_model_inference(model, model_name, t_in, s_in, device)
            T_DL_real = denormalize_transform(T_DL_norm, centroid_t, scale)

            # DL-only
            geo_rre, geo_rte = compute_registration_error(T_gt, T_DL_real)
            geo_errors.append((geo_rre, geo_rte))

            # DL + ICP
            T_Final_real = refine_registration_icp(
                pts_s_real, pts_t_real, T_DL_real,
                distance_threshold=args.icp_threshold,
                max_iter=args.icp_max_iter
            )
            icp_rre, icp_rte = compute_registration_error(T_gt, T_Final_real)
            icp_errors.append((icp_rre, icp_rte))

        g_metrics = calculate_metrics(geo_errors)
        i_metrics = calculate_metrics(icp_errors)

        results_summary[model_name] = {
            'DL_Only': g_metrics,
            'DL_ICP': i_metrics
        }

        del model
        torch.cuda.empty_cache()

    # 报告
    print("\n\n" + "★" * 108)
    print(f"🏆 {num_samples} 个样本评测报告 | angle={args.angle_range}°, trans={args.translation_range}")
    print(f"   成功标准: RTE < 0.01m 且 RRE < 1.0°")
    print("★" * 108)

    header = (
        f"{'模型名称':<18} | {'评测阶段':<10} | "
        f"{'RRE MSE(°)':<12} {'RRE MAE(°)':<12} | {'RTE MSE(m)':<12} {'RTE MAE(m)':<12} | {'成功率(%)':<8}"
    )
    print(header)
    print("-" * 108)

    for name, metrics in results_summary.items():
        g_r_mse, g_r_mae, g_t_mse, g_t_mae, g_succ = metrics['DL_Only']
        i_r_mse, i_r_mae, i_t_mse, i_t_mae, i_succ = metrics['DL_ICP']

        row_dl = (
            f"{name:<18} | {'仅深度学习':<10} | "
            f"{g_r_mse:<12.4f} {g_r_mae:<12.4f} | {g_t_mse:<12.4f} {g_t_mae:<12.4f} | {g_succ:<8.2f}"
        )
        row_icp = (
            f"{name:<18} | {'DL + ICP':<10} | "
            f"{i_r_mse:<12.4f} {i_r_mae:<12.4f} | {i_t_mse:<12.4f} {i_t_mae:<12.4f} | {i_succ:<8.2f}"
        )
        print(row_dl)
        print(row_icp)
        print("-" * 108)

    print("★" * 108 + "\n")


if __name__ == "__main__":
    main()