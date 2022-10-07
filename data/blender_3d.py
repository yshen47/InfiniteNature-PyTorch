import json
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import networkx as nx
from PIL import Image
import os, pickle


class PRNGMixin(object):
    """Adds a prng property which is a numpy RandomState which gets
    reinitialized whenever the pid changes to avoid synchronized sampling
    behavior when used in conjunction with multiprocessing."""

    @property
    def prng(self, seed=None):
        currentpid = os.getpid()
        if getattr(self, "_initpid", None) != currentpid:
            self._initpid = currentpid
            self._prng = np.random.RandomState(seed=seed) if seed is not None else np.random.RandomState()
        return self._prng


class Blender3dBase(Dataset, PRNGMixin):
    def __init__(self, split, dataset_dir=None, dataset=None, depth_range=None, image_resolution=None):
        self.split = split
        self.dataset_dir = dataset_dir
        self.src_num = 1
        os.makedirs(f"{self.dataset_dir}/cache", exist_ok=True)
        self.dataset = dataset
        self.image_resolution = image_resolution
        self.grids = []
        self.depth_range = depth_range
        self.cumulative_sum = [0]
        self.K = np.load(f"{self.dataset_dir}/K.npy")
        for grid_transform_path in sorted(Path(self.dataset_dir, self.split).glob("*")):
            print(grid_transform_path)
            with open(str(grid_transform_path / "transforms.json"), 'r') as f:
                curr_transform = json.load(f)
                g = self.build_graph_from_transform(curr_transform['frames'], grid_transform_path)
                self.grids.append(g)
                self.cumulative_sum.append(len(g.nodes) + self.cumulative_sum[-1])

    def build_graph_from_transform(self, transforms, grid_transform_path):
        g_path = f"{self.dataset_dir}/cache/{grid_transform_path.name[:-4]}_graph_{self.split}.txt"
        if os.path.exists(g_path):
           return pickle.load(open(g_path, 'rb'))
        g = nx.Graph()
        for i, transform in enumerate(transforms):
            c2w = np.array(transform['transform_matrix']) @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            w2c = np.linalg.inv(c2w)
            g.add_nodes_from([(i, {
                                    "R": w2c[:3, :3],
                                    "t": w2c[:3, 3],
                                    "position": c2w[:3, 3],
                                    "rgb_path": str(grid_transform_path / f"im_{i:05d}.png"),
                                    "depth_path": str(grid_transform_path / f"dm_{i:05d}.npy"),
                                    })])

        for i in range(len(transforms)-1):
            for j in range(i+1, len(transforms)):
                node_i = g.nodes[i]
                node_j = g.nodes[j]
                v = node_i['position'] - node_j['position']
                euclidean_dis = np.linalg.norm(v)
                if euclidean_dis <= 3: #TODO: to add as hyperparameters
                    g.add_edge(i, j, weight=euclidean_dis)
        pickle.dump(g, open(g_path, 'wb'))
        return g

    def __len__(self):
        return self.cumulative_sum[-1]

    def parse_idx(self, idx):
        for img_index, cur_cumsum in enumerate(self.cumulative_sum):
            if idx < self.cumulative_sum[img_index+1]:
                return img_index, idx - cur_cumsum

    def __getitem__(self, global_index):
        grid_id, idx = self.parse_idx(global_index)
        tgt_node = self.grids[grid_id].nodes[idx]
        tgt_neighbors = sorted(self.grids[grid_id][idx])
        if self.split == 'train':
            src_num = 1
            src_nodes = [self.grids[grid_id].nodes[tgt_neighbors[k]] for k in self.prng.choice(len(tgt_neighbors), src_num)]
        else:
            state = np.random.RandomState(seed=global_index)
            tgt_neighbors = np.array(tgt_neighbors)
            state.shuffle(tgt_neighbors)
            src_num = 1
            src_nodes = [self.grids[grid_id].nodes[k] for k in tgt_neighbors[:src_num]]
        img_dst = np.array(Image.open(tgt_node['rgb_path']))/127.5-1.0
        img_srcs = [np.array(Image.open(src_node['rgb_path']))/127.5-1.0 for src_node in src_nodes]

        dm_dst = np.load(tgt_node['depth_path'])#[..., None]
        dm_srcs = [np.load(src_node['depth_path']) for src_node in src_nodes]
        h, w = dm_dst.shape[:2]
        x = np.linspace(0, w - 1, w)
        y = np.linspace(0, h - 1, h)
        xs, ys = np.meshgrid(x, y)
        dm_dst = (dm_dst * self.K[0][0] / np.sqrt(self.K[0][0]**2 + (self.K[0][2] - ys - 0.5)**2 + (self.K[1][2] - xs - 0.5) ** 2))[..., None]
        dm_srcs = [(dm_src * self.K[0][0] / np.sqrt(self.K[0][0]**2 + (self.K[0][2] - ys - 0.5)**2 + (self.K[1][2] - xs - 0.5) ** 2))[..., None] for dm_src in dm_srcs]
        # for j in range(predicted_depth.shape[0]):
        #     for k in range(predicted_depth.shape[1]):
        #         predicted_depth[j, k] = predicted_depth[j, k] * K[0][0] / math.sqrt(K[0][0] ** 2 + (K[0][2] - j - 0.5) ** 2 + (K[1][2] - k - 0.5) ** 2)
        R_dst = tgt_node["R"]
        t_dst = tgt_node["t"]
        T_rels = []
        Ks = []
        K_invs = []
        T_tgt = np.eye(4)
        T_tgt[:3, :3] = R_dst
        T_tgt[:3, 3] = t_dst

        ## K
        h, w = img_dst.shape[:2]
        for src_node in src_nodes:
            R_src = src_node["R"]
            t_src = src_node["t"]
            T_src = np.eye(4)
            T_src[:3, :3] = R_src
            T_src[:3, 3] = t_src
            T_rel = T_tgt @ np.linalg.inv(T_src)
            T_rels.append(T_rel)
            Ks.append(self.K)

        example = {
            "Ks": Ks[0],
            "T_src2tgt": T_rels[0],
            "dst_img": img_dst,
            "src_img": img_srcs[0],
            "dst_disparity": 1/dm_dst,
            "src_disparity": 1/dm_srcs[0]
        }

        for k in example:
            example[k] = example[k].astype(np.float32)

        return example


class Blender3dTrain(Blender3dBase):
    def __init__(self, image_resolution=None, dataset_dir=None):
        super().__init__(split='train', dataset_dir=dataset_dir, image_resolution=image_resolution)
        self.image_resolution = image_resolution


class Blender3dValidation(Blender3dBase):
    def __init__(self, image_resolution=None, dataset_dir=None):
        super().__init__(split='val', dataset_dir=dataset_dir, image_resolution=image_resolution)
        self.image_resolution = image_resolution


class Blender3dTest(Blender3dBase):
    def __init__(self, image_resolution=None, dataset_dir=None):
        super().__init__(split='test', dataset_dir=dataset_dir, image_resolution=image_resolution)
        self.image_resolution = image_resolution

