import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, shape, num_samples) -> None:
        super().__init__()
        self.shape = shape
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        arr = torch.randn(self.shape) ** (5 / 4)
        for x in torch.randn(30000):
            arr *= x
        return arr


class BoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 1, 1)

    def forward(self, x):
        return self.layer(x)


def run(rank, world_size):
    cuda = False

    print("init rank", rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"
    torch.distributed.init_process_group(
        backend=("nccl" if cuda else "gloo"),
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    print("init rank done", rank)

    train = RandomDataset((3, 2048, 2048), 10000)
    train = DataLoader(train, batch_size=32, pin_memory=True, num_workers=2)

    model = BoringModel()
    ddp_model = DistributedDataParallel(model, device_ids=([rank] if cuda else None))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for i, batch in enumerate(train):
        print(i)
        prediction = ddp_model(batch)
        loss = torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    world_size = 2
    torch.multiprocessing.spawn(run, nprocs=world_size, args=(world_size,))
