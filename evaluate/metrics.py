from collections import defaultdict
from einops import rearrange
from torch import Tensor
import torch


class TopKAccumulator:
    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.total2 = 0
        self.total3 = 0
        self.metrics = defaultdict(float)

    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        """
        actual: Tensor of shape (B, D) - true items
        top_k: Tensor of shape (B, K, D) - top-k predicted items
        """
        B, D = actual.shape
        K = top_k.shape[1]

        # Expand actual to compare with top_k
        actual_exp = rearrange(actual, "b d -> b 1 d")  # shape (B, 1, D)
        match = (actual_exp == top_k).all(dim=-1)       # shape (B, K), bool
        # For Recall and NDCG
        match_indices = match.float().argmax(dim=1)     # shape (B,)
        match_exists = match.any(dim=1)                 # shape (B,)

        for k in self.ks:
            # --- Recall@k ---
            recall_at_k = match[:, :k].any(dim=1).float()  # [B]
            self.metrics[f"recall@{k}"] += recall_at_k.sum().item()

            # --- NDCG@k ---
            ndcg = torch.zeros(B, device=actual.device)
            valid = match_exists & (match_indices < k)
            ndcg[valid] = 1.0 / torch.log2(match_indices[valid].float() + 2.0)
            self.metrics[f"ndcg@{k}"] += ndcg.sum().item()

        # --- Hit Rate 按 D 各维度分别统计 ---
        for i in range(D):
            # 维度前缀匹配
            pos_match = (actual_exp[:, :, :i+1] == top_k[:, :, :i+1]).all(dim=-1)  # [B, K]
            match_found, rank = pos_match.max(dim=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_slice_{i+1}"] += (matched_rank < k).sum().item()

            # 精确第i维匹配
            pos_match = (actual_exp[:, :, i:i+1] == top_k[:, :, i:i+1]).all(dim=-1)  # [B, K]
            match_found, rank = pos_match.max(dim=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_pos_{i}"] += (matched_rank < k).sum().item()

        self.total += B

    def accumulate_match3(self, actual: Tensor, top_k: Tensor) -> None:
        B, D = actual.shape
        K = top_k.shape[1]

        actual_exp = rearrange(actual, "b d -> b 1 d")
        match = (actual_exp[:, :, :3] == top_k[:, :, :3]).all(dim=-1)  # 只匹配前两个维度
        match_indices = match.float().argmax(dim=1)
        match_exists = match.any(dim=1)

        for k in self.ks:
            recall_at_k = match[:, :k].any(dim=1).float()
            self.metrics[f"recall@{k}_match3"] += recall_at_k.sum().item()

            ndcg = torch.zeros(B, device=actual.device)
            valid = match_exists & (match_indices < k)
            ndcg[valid] = 1.0 / torch.log2(match_indices[valid].float() + 2.0)
            self.metrics[f"ndcg@{k}_match3"] += ndcg.sum().item()

        self.total2 += B

    def accumulate_valid_only(self, actual: Tensor, top_k: Tensor) -> None:
        B, D = actual.shape
        K = top_k.shape[1]

        actual_exp = rearrange(actual, "b d -> b 1 d")
        match = (actual_exp == top_k).all(dim=-1)
        match_indices = match.float().argmax(dim=1)
        match_exists = match.any(dim=1)

        # 只保留命中样本
        actual = actual[match_exists]
        match = match[match_exists]
        match_indices = match_indices[match_exists]
        B_valid = match.shape[0]

        for k in self.ks:
            recall_at_k = match[:, :k].any(dim=1).float()
            self.metrics[f"recall@{k}_valid"] += recall_at_k.sum().item()

            ndcg = torch.zeros(B_valid, device=actual.device)
            valid = match_indices < k
            ndcg[valid] = 1.0 / torch.log2(match_indices[valid].float() + 2.0)
            self.metrics[f"ndcg@{k}_valid"] += ndcg.sum().item()

        self.total3 += B_valid

    def reduce(self) -> dict:
        print(self.total)
        print(self.total2)
        print(self.total3)
        return {k: v / self.total for k, v in self.metrics.items()}

