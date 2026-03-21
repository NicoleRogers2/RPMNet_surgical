import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

class SurgicalRegistrationData(Dataset):
    def __init__(
        self,
        data_root="./learning3d/data//modelnet40_ply_hdf5_2048",
        partition='test',
        template_points=2048,
        source_points=1024,
        angle_range=180,
        translation_range=2.0, 
        noise_sigma=0.005,
        coverage_ratio=0.90,  # 划线覆盖面积比例
    ):
        super().__init__()
        self.template_points = template_points
        self.source_points = source_points
        self.angle_range_rad = angle_range * (np.pi / 180)
        self.translation_range = translation_range
        self.noise_sigma = noise_sigma
        self.coverage_ratio = coverage_ratio

        # 加载 HDF5 数据
        self.data, self.labels = self._load_data(data_root, partition)

    def _load_data(self, data_root, partition):
        data, labels = [], []
        h5_files = glob.glob(os.path.join(data_root, f'ply_data_{partition}*.h5'))
        if not h5_files:
            raise FileNotFoundError(f"未在 {data_root} 找到 {partition} 的 h5 数据文件。")
        
        for h5_name in h5_files:
            with h5py.File(h5_name, 'r') as f:
                data.append(f['data'][:].astype('float32'))
                labels.append(f['label'][:].astype('int64'))
                
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        return data, labels

    def __len__(self):
        return self.data.shape[0]

    def _generate_transform(self):
        anglex = np.random.uniform(-1, 1) * self.angle_range_rad
        angley = np.random.uniform(-1, 1) * self.angle_range_rad
        anglez = np.random.uniform(-1, 1) * self.angle_range_rad
        translation = np.array([
            np.random.uniform(-self.translation_range, self.translation_range),
            np.random.uniform(-self.translation_range, self.translation_range),
            np.random.uniform(-self.translation_range, self.translation_range),
        ], dtype=np.float32)

        rotation = Rotation.from_euler('zyx', [anglez, angley, anglex])
        R = rotation.as_matrix().astype(np.float32)
        igt = np.eye(4, dtype=np.float32)
        igt[:3, :3] = R
        igt[:3, 3] = translation

        return R, translation, igt

    def _random_half_cut(self, points):
        """随机切掉一半，保留另一半"""
        centroid = np.mean(points, axis=0)
        normal = np.random.randn(3)
        normal /= np.linalg.norm(normal)
        
        dots = np.dot(points - centroid, normal)
        keep_mask = dots > 0
        half_points = points[keep_mask]
        
        if len(half_points) < len(points) * 0.2:
            indices = np.argsort(dots)
            half_points = points[indices[len(points)//2:]]
            
        return half_points

    def _generate_continuous_probe_path(self, points, coverage_ratio, num_source_pts):
        """
        生成单段不分叉、紧贴表面的划线，并且保证输出是非同源的点云。
        """
        N = len(points)
        target_len = int(N * coverage_ratio)

        visited = np.zeros(N, dtype=bool)
        
        # 为了避免一开始就陷入死胡同，从物体的边缘（极大值点）开始游走
        start_idx = np.argmax(points[:, 0]) 
        visited[start_idx] = True
        path = [start_idx]

        curr_idx = start_idx
        # 初始运动方向，随意给一个
        current_dir = np.array([1.0, 0.0, 0.0]) 

        # 维护一个未访问点的索引池，加速查找
        unvisited_idx = np.where(~visited)[0]

        for _ in range(target_len - 1):
            curr_pt = points[curr_idx]

            # 1. 计算当前点到所有未访问点的向量和距离
            vecs = points[unvisited_idx] - curr_pt
            dists = np.linalg.norm(vecs, axis=1)

            # 2. 找到空间上绝对最近的 K 个未访问点 (限制在局部表面，防止飞跃空气)
            k = min(15, len(unvisited_idx))
            if k == 0: break
            
            idx_k = np.argpartition(dists, k - 1)[:k]
            dists_k = dists[idx_k]
            vecs_k = vecs[idx_k]

            # 3. 在这 K 个局部点中，结合“距离”和“运动惯性”打分
            # 鼓励探针沿着先前的方向平滑滑动，而不是在原地来回无规则跳跃
            norms_k = dists_k + 1e-8
            vecs_norm_k = vecs_k / norms_k[:, None]
            cos_theta_k = np.dot(vecs_norm_k, current_dir)

            # 归一化距离惩罚
            d_min, d_max = dists_k.min(), dists_k.max()
            if d_max > d_min:
                dists_k_norm = (dists_k - d_min) / (d_max - d_min)
            else:
                dists_k_norm = np.zeros_like(dists_k)

            # 综合 Score: 距离越近越好(小)，方向越顺越好(cos大)
            scores = dists_k_norm - 0.8 * cos_theta_k

            # 选择最佳下一步
            best_k = np.argmin(scores)
            chosen_unvisited_idx = idx_k[best_k]
            next_idx = unvisited_idx[chosen_unvisited_idx]

            # 4. 更新运动方向惯性
            step_vec = points[next_idx] - curr_pt
            step_dir = step_vec / (np.linalg.norm(step_vec) + 1e-8)
            current_dir = 0.5 * current_dir + 0.5 * step_dir  # 惯性平滑
            current_dir /= np.linalg.norm(current_dir)

            # 5. 更新状态
            curr_idx = next_idx
            visited[curr_idx] = True
            path.append(curr_idx)

            # O(1) 的时间复杂度从未访问池中移除该点
            unvisited_idx[chosen_unvisited_idx] = unvisited_idx[-1]
            unvisited_idx = unvisited_idx[:-1]

        path_points = points[path]

        # =================================================================
        # 核心：等弧长连续重采样 (Arc-length Resampling)
        # 这一步保证了生成的数据与原模型是“非同源”的！
        # 插值出的新点云位于多边形线段上，拥有全新的坐标，模拟探针的真实高频采样
        # =================================================================
        if len(path_points) >= 2:
            diffs = np.diff(path_points, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
            total_length = cum_length[-1]

            if total_length > 1e-8:
                target_lengths = np.linspace(0, total_length, num_source_pts)
                source_pts = np.zeros((num_source_pts, 3), dtype=np.float32)
                for dim in range(3):
                    source_pts[:, dim] = np.interp(target_lengths, cum_length, path_points[:, dim])
                return source_pts

        idx = np.random.choice(len(path_points), num_source_pts, replace=True)
        return path_points[idx].copy()

    def __getitem__(self, index):
        full_points = self.data[index]

        # ---- Target (Template): 完整的物体模型 ----
        if len(full_points) > self.template_points:
            t_idx = np.random.choice(len(full_points), self.template_points, replace=False)
            template = full_points[t_idx].copy()
        else:
            template = full_points[:self.template_points].copy()

        # ---- Step 1: 切掉一半 ----
        half_template = self._random_half_cut(template)

        # ---- Step 2: 生成连续不分叉、非同源的探针划线 ----
        source_raw = self._generate_continuous_probe_path(half_template, self.coverage_ratio, self.source_points)

        # ---- 施加探针测量噪声 (进一步增强非同源和现实感) ----
        if self.noise_sigma > 0:
            source_raw += np.random.normal(0, self.noise_sigma, source_raw.shape).astype(np.float32)

        # ---- 施加大范围刚体变换 ----
        R, t, igt = self._generate_transform()
        source_transformed = (R @ source_raw.T).T + t

        # 转 Tensor
        template_t = torch.from_numpy(template).float()
        source_t = torch.from_numpy(source_transformed.astype(np.float32)).float()
        igt_t = torch.from_numpy(igt).float()

        return template_t, source_t, igt_t


# ===================== 可视化检查代码 =====================
if __name__ == '__main__':
    # 运行此脚本直观检查划线
    dataset = SurgicalRegistrationData(
        data_root="./learning3d/data//modelnet40_ply_hdf5_2048",
        partition='test',
        template_points=2048,
        source_points=2048,   # 探针等弧长重采样点数
        angle_range=90,      
        translation_range=1.0,
        noise_sigma=0.000,    # 暂关噪声，方便观察线条连续性
        coverage_ratio=0.90,
    )

    print(f"数据集大小: {len(dataset)}")

    import open3d as o3d

    for i in range(3):
        template, source, igt = dataset[i]
        t_np = template.numpy()
        s_np = source.numpy()

        R_gt = igt[:3, :3].numpy()
        t_gt = igt[:3, 3].numpy()
        source_unwarped = (np.linalg.inv(R_gt) @ (s_np - t_gt).T).T

        t_pcd = o3d.geometry.PointCloud()
        t_pcd.points = o3d.utility.Vector3dVector(t_np)
        t_pcd.paint_uniform_color([0.8, 0.8, 0.8])
        
        s_pcd = o3d.geometry.PointCloud()
        s_pcd.points = o3d.utility.Vector3dVector(source_unwarped)
        s_pcd.paint_uniform_color([0, 0, 1])

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 4.0  

        print(f"展示第 {i} 个样本：连续不分叉、非同源重采样划线。")
        o3d.visualization.draw_geometries([t_pcd, s_pcd], window_name=f"Sample {i} - Continuous Non-homologous Path")