import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

def emp_q(Y, tau_hat):
    return np.quantile(Y, tau_hat)

def grad_ecdf(theta, Y_sorted, tau_hat):
    counts = np.searchsorted(Y_sorted, theta, side='right')
    return counts / Y_sorted.size - tau_hat

def project_isotonic(theta):
    return np.maximum.accumulate(theta)

def update(Y, theta, tau_hat, iters=500, lr=0.05):
    Y_sorted = np.sort(Y)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8

    for t in range(1, iters + 1):
        g = grad_ecdf(theta, Y_sorted, tau_hat)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
        theta = project_isotonic(theta)
    return theta

def quantile_pdf(theta, tau_hat, x_grid):
    return gaussian_kde(theta, bw_method=0.3)(x_grid)


# --- Setup ---
K = 50
tau_hat = (np.arange(K) + 0.5) / K
x_grid = np.linspace(-4, 4, 400)
true_pdf = norm.pdf(x_grid)

sample_sizes = [10, 100, 10000]
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, S in enumerate(sample_sizes):
    Y = np.random.normal(size=S)
    theta0 = np.full(K, Y.mean(), dtype=float)
    theta = update(Y, theta0, tau_hat)

    pdf_piecewise = quantile_pdf(np.sort(theta), tau_hat, x_grid)

    axs[i].plot(x_grid, true_pdf, label='True Normal PDF', lw=2)
    axs[i].plot(x_grid, pdf_piecewise, linestyle='--', lw=2, label='Quantile-based PDF')
    axs[i].set_title(f'Snapshot with S = {S}')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('Density')
    axs[i].legend()

plt.suptitle(f'Quantile Approximation Convergence (K={K})', fontsize=14)
plt.tight_layout()
plt.show()
