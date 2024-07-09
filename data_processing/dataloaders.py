from monai.data import DataLoader
import torch


class DatasetFlattner(torch.utils.data.Dataset):
    def __init__(self, batch_data: list, max_elements: int, n_patches: int):
        self.batch_data = batch_data  # len(batch_data) = n_batches
        self.max_elements = max_elements  # number of patches in the whole dataset, needed to handle uneven? cases
        self.n_patches = n_patches

    def __len__(self):
        return self.max_elements

    def __getitem__(self, idx):
        data = dict()
        for key, val in self.batch_data[idx // self.n_patches].items():
            if isinstance(val, torch.Tensor):
                data[key] = val[idx % self.n_patches]
            # else:
            #     data[key] = val
        return data


class ShuffleDataLoader(torch.utils.data.DataLoader):
    def __init__(self, monai_dataset, n_patches, monai_num_workers, *args, **kwargs):
        self.monai_dataset = monai_dataset
        self.dataset = torch.zeros(len(monai_dataset) * n_patches)
        self.n_patches = n_patches
        self.monai_num_workers = monai_num_workers
        self.buffer_size = min(monai_num_workers, len(monai_dataset))
        super(ShuffleDataLoader, self).__init__(torch.zeros(len(monai_dataset) * n_patches), *args, **kwargs)

    def __setattr__(self, attr, val):
        if attr in ('batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset', 'persistent_workers', 'num_workers'):
            super(torch.utils.data.DataLoader, self).__setattr__(attr, val)
        else:
            super(ShuffleDataLoader, self).__setattr__(attr, val)

    def _get_iterator(self):
        monai_dataloader = DataLoader(self.monai_dataset, batch_size=self.buffer_size, shuffle=True,
                                      num_workers=self.monai_num_workers)
        self.dataset = DatasetFlattner(batch_data=[_ for _ in monai_dataloader],
                                       max_elements=len(self.monai_dataset),
                                       n_patches=self.n_patches * self.buffer_size)
        return super()._get_iterator()
