import torch

def AdamW(model, learning_rate=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer