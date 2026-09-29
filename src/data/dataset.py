from __future__ import annotations


class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


def load_idx(path: str):
    raise NotImplementedError
