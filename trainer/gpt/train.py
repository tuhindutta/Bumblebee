from .estimate_loss import estimate_loss
from .adam_optimizer import AdamW
from .save_checkpoints import save_checkpoint




def Trainer(model,
            train_loader,
            val_loader,
            max_iters = 1,
            eval_interval = 100,
            eval_iters = 200,
            learning_rate = 3e-4):
    
    optimizer = AdamW(model, learning_rate)

    train_losses = []
    val_losses = []

    for iteration in range(max_iters):
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # Evaluation
            if batch_idx % eval_interval == 0 or batch_idx == len(train_loader) - 1:
                losses = estimate_loss(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    eval_iters=min(eval_iters, len(val_loader))
                )
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])

                print(
                    f"iteration {iteration} / step {batch_idx}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )

            # Training step
            logits, loss = model(x_batch, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=iteration,
            loss=loss.item(),
            file_path=f"./gpt_checkpoints/checkpoint_{iteration}.pth"
        )

    return model, train_losses, val_losses