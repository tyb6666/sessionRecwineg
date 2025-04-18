import gin
import os
import random
import torch

from data.amazon import AmazonReviews
from data.ml1m import RawMovieLens1M
from data.ml32m import RawMovieLens32M
from data.schemas import SeqBatch
from enum import Enum
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional

PROCESSED_MOVIE_LENS_SUFFIX = "/processed/data.pt"


@gin.constants_from_enum
class RecDataset(Enum):
    AMAZON = 1
    ML_1M = 2
    ML_32M = 3


DATASET_NAME_TO_RAW_DATASET = {
    RecDataset.AMAZON: AmazonReviews,
    RecDataset.ML_1M: RawMovieLens1M,
    RecDataset.ML_32M: RawMovieLens32M
}


DATASET_NAME_TO_MAX_SEQ_LEN = {
    RecDataset.AMAZON: 20,
    RecDataset.ML_1M: 200,
    RecDataset.ML_32M: 200
}


class ItemData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        session: int = 1,
        force_process: bool = False,
        dataset: RecDataset = RecDataset.ML_1M,
        train_test_split: str = "all",
        **kwargs
    ) -> None:
        
        raw_dataset_class = DATASET_NAME_TO_RAW_DATASET[dataset]
        max_seq_len = DATASET_NAME_TO_MAX_SEQ_LEN[dataset]

        raw_data = raw_dataset_class(root=root, *args, session = session, **kwargs)
        
        processed_data_path = raw_data.processed_paths[0]
        if not os.path.exists(processed_data_path) or force_process:
            raw_data.process(max_seq_len=max_seq_len)
        
        if train_test_split == "train":
            filt = raw_data.data["item"]["is_train"]
        elif train_test_split == "eval":
            filt = ~raw_data.data["item"]["is_train"]
        elif train_test_split == "all":
            filt = torch.ones_like(raw_data.data["item"]["x"][:,0], dtype=bool)

        self.item_data, self.item_text = raw_data.data["item"]["x"][filt], raw_data.data["item"]["text"][filt]

    def __len__(self):
        return self.item_data.shape[0]

    def __getitem__(self, idx):
        item_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        x = self.item_data[idx, :768]
        return SeqBatch(
            user_ids=-1 * torch.ones_like(item_ids.squeeze(0)),
            ids=item_ids,
            ids_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            x=x,
            x_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            seq_mask=torch.ones_like(item_ids, dtype=bool)
        )


class SeqData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        session: int = 1,
        negnum : int = 0,
        is_train: bool = True,
        subsample: bool = False,
        force_process: bool = False,
        dataset: RecDataset = RecDataset.ML_1M,
        **kwargs
    ) -> None:
        
        assert (not subsample) or is_train, "Can only subsample on training split."

        raw_dataset_class = DATASET_NAME_TO_RAW_DATASET[dataset]
        max_seq_len = DATASET_NAME_TO_MAX_SEQ_LEN[dataset]

        raw_data = raw_dataset_class(root=root, *args, session = session, **kwargs)
        self.session = session

        processed_data_path = raw_data.processed_paths[0]
        if not os.path.exists(processed_data_path) or force_process:
            raw_data.process(max_seq_len=max_seq_len)

        split = "train" if is_train else "test"
        self.subsample = subsample
        self.sequence_data = raw_data.data[("user", "rated", "item")]["history"][split]
        # print(self.sequence_data)
        # print(self.sequence_data["itemId"].shape)
        # print(self.sequence_data["itemId_fut"].shape)
        # input()
        if not self.subsample:
            self.sequence_data["itemId"] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(l[-max_seq_len:]) for l in self.sequence_data["itemId"]],
                batch_first=True,
                padding_value=-1
            )

        self._max_seq_len = max_seq_len
        self.item_data = raw_data.data["item"]["x"]
        # print(self.item_data)
        # print(self.item_data.shape)
        # print(self.item_data.shape)
        # input()
        self.split = split
        self.negnum = negnum
    
    
    @property
    def max_seq_len(self):
        return self._max_seq_len

    def __len__(self):
        return self.sequence_data["userId"].shape[0]
  
    def __getitem__(self, idx):
        user_ids = self.sequence_data["userId"][idx]
        
        if self.subsample:
            seq = torch.cat([self.sequence_data["itemId"][idx][: (self.sequence_data["itemId"][idx] != -1).nonzero(as_tuple=True)[0][-1] + 1], self.sequence_data["itemId_fut"][idx]])
            max_start = max(0, len(seq) - self.session - 3)
            start_idx = random.randint(0, max_start)
            end_idx = random.randint(start_idx+3, start_idx+self.max_seq_len+ self.session)
            sample = seq[start_idx:end_idx]
            
            input_part = sample[:-self.session] if len(sample) > self.session else []
            item_ids = torch.tensor(input_part.tolist() + [-1] * (self.max_seq_len - len(input_part)))
            fut_part = sample[-self.session:] if len(sample) >= self.session else sample[-1:] + [-1] * (self.session - 1)
            item_ids_fut = torch.tensor(fut_part.tolist() + [-1] * (self.session - len(fut_part)))

        else:
            item_ids = self.sequence_data["itemId"][idx]
            item_ids_fut = self.sequence_data["itemId_fut"][idx]

            # ----------------- 负采样操作 -----------------
        flag_neg = 100000
        if self.negnum > 0:
            history_ids = torch.cat([self.sequence_data["itemId"][idx], self.sequence_data["itemId_fut"][idx]])
            history_ids = history_ids[history_ids >= 0].unique().tolist()  # 去重 & 去 -1
            all_item_ids = set(range(len(self.item_data)))
            candidate_ids = list(all_item_ids - set(history_ids))
            neg_samples = random.sample(candidate_ids, min(self.negnum, len(candidate_ids)))
            neg_samples = [item_id + flag_neg for item_id in neg_samples]  # 加偏移区分

            neg_samples_tensor = torch.tensor(neg_samples, dtype=item_ids.dtype, device=item_ids.device)

            # 拼接正负 ID
            item_ids = torch.cat([item_ids, neg_samples_tensor])

            item_ids_raw = item_ids.clone()
            item_ids_for_x = item_ids_raw.clone()
            item_ids_for_x[item_ids_for_x >= flag_neg] -= flag_neg

            assert (item_ids >= -1).all(), "Invalid movie id found"
            x = self.item_data[item_ids_for_x, :768]
            x[item_ids == -1] = -1

            x_fut = self.item_data[item_ids_fut, :768]
            x_fut[item_ids_fut == -1] = -1

        else:

            assert (item_ids >= -1).all(), "Invalid movie id found"
            x = self.item_data[item_ids, :768]
            x[item_ids == -1] = -1

            x_fut = self.item_data[item_ids_fut, :768]
            x_fut[item_ids_fut == -1] = -1


        return SeqBatch(
            user_ids=user_ids,
            ids=item_ids,
            ids_fut=item_ids_fut,
            x=x,
            x_fut=x_fut,
            seq_mask=(item_ids >= 0)
        )


if __name__ == "__main__":
    dataset = ItemData("dataset/amazon", dataset=RecDataset.AMAZON, split="beauty", force_process=True)
    dataset[0]
    import pdb; pdb.set_trace()
