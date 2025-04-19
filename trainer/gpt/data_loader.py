import torch
torch.manual_seed(3647)
from typing import Tuple
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index:index + self.block_size]
        y = self.data[index + 1:index + self.block_size + 1]
        return x, y
    

def split_data(tokenized_text_data, train_split_pct:float = 0.9):
    data = torch.tensor(tokenized_text_data, dtype=torch.long)
    split_index = int(train_split_pct*len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]
    return train_data, val_data


def get_dataloaders(
        data: torch.Tensor,
        block_size: int,
        batch_size: int,
        device: torch.device,
        train_split_pct: float = 0.9
) -> Tuple[DataLoader, DataLoader]:
    
    train_data, val_data = split_data(data, train_split_pct)

    if len(train_data) <= block_size:
        raise ValueError(f"Train data too small ({len(train_data)}) for block_size={block_size}")
    if len(val_data) <= block_size:
        raise ValueError(f"Validation data too small ({len(val_data)}) for block_size={block_size}")

    train_dataset = TextDataset(train_data.to(device), block_size)
    val_dataset = TextDataset(val_data.to(device), block_size)

    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory      
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory
    )

    return train_loader, val_loader




# def get_dataloaders(
#         train_data: torch.Tensor,
#         val_data: torch.Tensor,
#         block_size: int,
#         batch_size: int,
#         device: torch.device
# ) -> Tuple[DataLoader, DataLoader]:
#     train_dataset = TextDataset(train_data.to(device), block_size)
#     val_dataset = TextDataset(val_data.to(device), block_size)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#     )

#     return train_loader, val_loader