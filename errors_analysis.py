import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


dirs = [f"new_data/{d}" for d in os.listdir('new_data')]
gt = np.concatenate([np.load(f"{d}/true.npy") for d in dirs]) * 720
preds = np.concatenate([np.load(f"{d}/preds.npy") for d in dirs]) * 720


error = gt - preds

plt.hist(error[:, 0], bins=100)
plt.xlabel("relative error x")
plt.ylabel("n samples")
plt.savefig("figures/error_hist_x.png")

plt.hist(error[:, 1], bins=100)
plt.xlabel("relative error y")
plt.ylabel("n samples")
plt.savefig("figures/error_hist_y.png")


# for d in dirs:
#     gt = np.load(f"{d}/true.npy")
#     preds = np.load(f"{d}/preds.npy")
#     error = np.sqrt(np.square(preds - gt).sum(-1)).mean()
#     print(f"RMSE {error}")
#     # preds[:, 0] = np.convolve(preds[:, 0], np.ones(5) / 5, 'valid')
#     # preds[:, 1] = np.convolve(preds[:, 1], np.ones(5) / 5, 'valid')
#     preds = np.stack([np.convolve(preds[:, i], np.ones(9) / 9, 'valid') for i in (0, 1)], axis=-1)
#     error = np.sqrt(np.square(preds - gt[4:-4]).sum(-1)).mean()
#     print(f"RMSE after filter {error}")


# One Euro Filter
def smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, dt, x0, dx0=0.0, min_cutoff=0.00001, beta=20,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""

        self.min_cutoff = np.ones_like(x0) * min_cutoff
        self.beta = np.ones_like(x0) * beta
        self.d_cutoff = np.ones_like(x0) * d_cutoff
        # Previous values.
        self.x_prev = np.array(x0)
        self.dx_prev = np.ones_like(x0) * dx0
        self.dt = dt

    def __call__(self, x):
        """Compute the filtered signal."""
        t_e = self.dt

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat


for d in dirs:
    print(d)
    gt = np.load(f"{d}/true.npy") * np.array((1024, 2448))
    preds = np.load(f"{d}/preds.npy") * np.array((1024, 2448))
    # filtered = [preds[0]]
    # one_euro_x = OneEuroFilter(1 / 40, preds[0, 0])
    # one_euro_y = OneEuroFilter(1 / 40, preds[0, 1])
    # for pr in tqdm(preds[1:], total=len(preds[1:])):
    #     filtered.append(np.array((
    #         one_euro_x(pr[0]), one_euro_y(pr[1])
    #     )))
    # filtered = np.array(filtered)
    # np.save(f"{d}/filtered.npy", filtered)
    filtered = np.stack([np.convolve(preds[:, i], np.ones(9) / 9, 'valid') for i in (0, 1)], axis=-1)
    error = np.sqrt(np.square(preds - gt).sum(-1)).mean()
    print(f"RMSE {error}")
    plt.figure()
    plt.plot(gt[:,0], 'b')
    plt.plot(filtered[:,0], 'r')
    plt.scatter(np.arange(0, preds.shape[0]), preds[:,0], s=1)
    plt.savefig(f'figures/{d[-1]}_gt_pred_x.png', dpi=600)
    plt.figure()
    plt.plot(gt[:,1], 'b', label='real trajectory')
    # plt.plot(filtered[:,1], 'r')
    # plt.scatter(np.arange(0, preds.shape[0]), preds[:,1], s=1, )
    plt.plot(np.arange(0, preds.shape[0]), preds[:,1], label='predicted trajectory', color='r')
    plt.legend()
    plt.savefig(f'figures/{d[-1]}_gt_pred_y.png', dpi=600)
    # error = np.sqrt(np.square(preds - gt[4:-4]).sum(-1)).mean()
    # print(f"RMSE after filter {error}")

    # break