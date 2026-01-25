from sklearn.neighbors import KDTree
import numpy as np
import torch

class RetrievalPolicy:
    def __init__(self, model, traj_lib):

        self.model = model
        self.model.eval()

        self.library = traj_lib

        # --- 离线构造索引 Offline Indexing --- 

        # 基于起点关节角度构造几何 KD-Tree
        start_joints = [item['start_joint'] for item in self.library]
        self.geo_tree = KDTree(np.array(start_joints))

        # 构造语义嵌入库
        all_trajs = torch.tensor(np.array( item['full_traj'] for item in self.library))
        with torch.no_grad():
            self.traj_embeds = self.model.encode_traj(all_trajs)


    def get_action(self, cur_img, cur_joint, k_geo=5, k_sem=1):
        """
            cur_img: 当前相机图片
            cur_joint: 当前机械臂关节角度
            k_geo: 选取离当前末端位姿最近的 k_geo 簇轨迹
            k_sem: 语义选择后
        """

        # 找到最近的轨迹
        dist, indices = self.geo_tree.query(cur_joint.reshape(1,-1),k=k_geo)
        candidate_indices = indices[0]

        # 获得候选者的语义 Embedding
        candidate_embeds = self.traj_embeds[candidate_indices]

        # 计算当前的图片语义 Embedding
        with torch.no_grad():
            # extend to batch dim
            cur_img_tensor = preprocess_image(cur_img).unsqueeze(0)
            query_embed = self.model.encode_image(cur_img_tensor)

        # similarity
        # [1,512] @ [512, k_geo] -> [1,k_geo]
        similarities = (query_embed @ candidate_embeds.t()).squeeze(0)

        # 选择最佳匹配
        best_match_idx_in_candidates = torch.argmax(similarities).item()
        final_idx = candidate_indices[best_match_idx_in_candidates]

        best_traj = self.library[final_idx]['full_traj']
        
        