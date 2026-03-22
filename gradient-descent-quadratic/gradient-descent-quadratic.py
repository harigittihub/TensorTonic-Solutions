def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    x = float(x0)   # ensure float

    for _ in range(steps):
        x = x - lr * (2 * a * x + b)

    return x