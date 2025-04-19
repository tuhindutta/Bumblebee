import os
import torch
from .estimate_loss import estimate_loss
from .save_checkpoints import save_checkpoint


def Trainer(model,
            device,
            checkpoint_parent_path: str,
            optimizer,
            train_loader,
            val_loader,
            max_iters=1,
            eval_interval=100,
            eval_iters=200):

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for iteration in range(max_iters):
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # Move to device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Evaluation
            if batch_idx % eval_interval == 0 or batch_idx == len(train_loader) - 1:
                model.eval()
                with torch.no_grad():
                    losses = estimate_loss(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        eval_iters=min(eval_iters, len(val_loader))
                    )
                model.train()

                train_losses.append(losses['train'])
                val_losses.append(losses['val'])

                print(
                    f"iteration {iteration} / step {batch_idx}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )

                # Save best checkpoint
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=iteration,
                        loss=losses['val'],
                        file_path=os.path.join(checkpoint_parent_path, f"checkpoint_{iteration}.pth")
                    )

            # Training step
            logits, loss = model(x_batch, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model, train_losses, val_losses
