import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    """
    Returns: tuple of (updated_weights_list, last_avg_gradient_list)
    """
    w = torch.tensor(w_init, dtype=torch.float32, requires_grad=True)
    last_avg_grad = None

    for i, (x_list, target) in enumerate(micro_batches):
        x = torch.tensor(x_list, dtype=torch.float32)
        t = torch.tensor(target, dtype=torch.float32)
        pred = torch.dot(w, x)
        loss = (pred - t) ** 2
        loss.backward()

        if (i + 1) % accum_steps == 0:
            last_avg_grad = w.grad.clone() / accum_steps
            with torch.no_grad():
                w -= lr * last_avg_grad
            w.grad.zero_()

    return w.detach().tolist(), last_avg_grad.tolist()
