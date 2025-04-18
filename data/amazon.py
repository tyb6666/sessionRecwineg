import gzip
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import polars as pl
import torch

from collections import defaultdict
from data.preprocessing import PreprocessingMixin
from torch_geometric.data import download_google_url
from torch_geometric.data import extract_zip
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from typing import Callable
from typing import List
from typing import Optional


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


class AmazonReviews(InMemoryDataset, PreprocessingMixin):
    gdrive_id = "1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G"
    gdrive_filename = "P5_data.zip"

    def __init__(
        self,
        root: str,
        split: str,  # 'beauty', 'sports', 'toys'
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        session: int = 1,
    ) -> None:
        self.split = split
        self.session = session
        super(AmazonReviews, self).__init__(
            root, transform, pre_transform, force_reload
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)
    
    @property
    def raw_file_names(self) -> List[str]:
        return [self.split]
    
    @property
    def processed_dir(self) -> str:
        # 根据 session 创建不同的子目录
        return os.path.join(self.root, 'processed', f'session_{self.session}')
    
    @property
    def processed_file_names(self) -> str:
        return f'data_{self.split}_s{self.session}.pt'

    
    def download(self) -> None:
        # path = download_google_url(self.gdrive_id, self.root, self.gdrive_filename)
        path = "/data/P5_data.zip"
        extract_zip(path, self.root)
        # os.remove(path)
        folder = osp.join(self.root, 'data')
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)
    
    def _remap_ids(self, x):
        return x - 1

    def train_test_split(self, max_seq_len=20, session: int = 1):
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        user_ids = []

        with open(os.path.join(self.raw_dir, self.split, "sequential_data.txt"), "r") as f:
            for line in f:
                parsed_line = list(map(int, line.strip().split()))
                user_ids.append(parsed_line[0])
                items = [self._remap_ids(id) for id in parsed_line[1:]]
                # print(f"items:\n{items}")

                if len(items) < session + 1:
                    continue  # 不够 session 个 future item 的跳过
                
                # train set: all but last `session` items
                train_hist = items[:-(session + 1)]
                train_target = items[-(session + 1):-1]
                sequences["train"]["itemId"].append(train_hist)
                sequences["train"]["itemId_fut"].append(train_target)

                # eval set: last `session + 1` items, history is [-session-2:-2], target is [-2:-1]
                eval_hist = items[-(max_seq_len + session + 1):-(session + 1)]
                eval_hist_padded = eval_hist + [-1] * (max_seq_len - len(eval_hist))
                eval_target = items[-(session + 1):-1]
                sequences["eval"]["itemId"].append(eval_hist_padded)
                sequences["eval"]["itemId_fut"].append(eval_target)

                # test set: similar, predicting the final item(s)
                test_hist = items[-(max_seq_len + session): -session]
                test_hist_padded = test_hist + [-1] * (max_seq_len - len(test_hist))
                test_target = items[-session:]
                sequences["test"]["itemId"].append(test_hist_padded)
                sequences["test"]["itemId_fut"].append(test_target)

                # print("sequences[train][itemId]:")
                # print(sequences["train"]["itemId"])
                # print("sequences[train][itemId_fut]:")
                # print(sequences["train"]["itemId_fut"])

                # print("sequences[test][itemId]:")
                # print(sequences["test"]["itemId"])
                # print("sequences[test][itemId_fut]:")
                # print(sequences["test"]["itemId_fut"])

        for sp in splits:
            sequences[sp]["userId"] = user_ids
            sequences[sp] = pl.from_dict(sequences[sp])
        return sequences
    
    def process(self, max_seq_len=20) -> None:
        data = HeteroData()

        with open(os.path.join(self.raw_dir, self.split, "datamaps.json"), 'r') as f:
            data_maps = json.load(f)    

        # Construct user sequences
        sequences = self.train_test_split(max_seq_len=max_seq_len, session = self.session)
        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"])
            for k, v in sequences.items() 
        }
        # Compute item features
        asin2id = pd.DataFrame([{"asin": k, "id": self._remap_ids(int(v))} for k, v in data_maps["item2id"].items()])
        item_data = (
            pd.DataFrame([
                meta for meta in
                parse(path=os.path.join(self.raw_dir, self.split, "meta.json.gz"))
            ])
            .merge(asin2id, on="asin")
            .sort_values(by="id")
            .fillna({"brand": "Unknown"})
        )

        sentences = item_data.apply(
            lambda row:
                "Title: " +
                str(row["title"]) + "; " +
                "Brand: " +
                str(row["brand"]) + "; " +
                "Categories: " +
                str(row["categories"][0]) + "; " + 
                "Price: " +
                str(row["price"]) + "; ",
            axis=1
        )
        
        item_emb = self._encode_text_feature(sentences)
        data['item'].x = item_emb
        data['item'].text = np.array(sentences)

        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05

        # 在保存时，确保创建相应的目录
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        self.save([data], self.processed_paths[0])
        



#if __name__ == "__main__":
#    AmazonReviews("dataset/amazon")
