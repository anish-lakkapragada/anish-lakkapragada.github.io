import numpy as np
import matplotlib.pyplot as plt

def stump_booster(X, y, T):
    """
    AdaBoost with decision stumps (including polarity).
    Returns:
      alphas:      array of length T of stump weights
      feat_inds:   array of length T of feature indices (0 or 1)
      thresholds:  array of length T of thresholds
      polarities:  array of length T of polarities (±1)
    """
    m, n = X.shape
    w = np.ones(m) / m

    alphas = []
    feat_inds = []
    thresholds = []
    polarities = []

    for t in range(T):
        best_err = np.inf
        best_j = None
        best_s = None
        best_d = None
        best_pred = None

        # Search over features, thresholds, and both polarities
        for j in range(n):
            vals = np.unique(X[:, j])
            thresh_cands = (vals[:-1] + vals[1:]) / 2.0

            for s in thresh_cands:
                for d in (+1, -1):
                    # stump prediction with polarity
                    pred = d * np.sign(X[:, j] - s)
                    pred[pred == 0] = d
                    err = np.sum(w * (pred != y))

                    if err < best_err:
                        best_err = err
                        best_j = j
                        best_s = s
                        best_d = d
                        best_pred = pred.copy()

        # compute stump weight
        eps = best_err
        alpha = 0.5 * np.log((1 - eps) / eps)

        # record stump parameters
        alphas.append(alpha)
        feat_inds.append(best_j)
        thresholds.append(best_s)
        polarities.append(best_d)

        # update example weights
        w *= np.exp(-alpha * y * best_pred)
        w /= np.sum(w)

    return (
        np.array(alphas),
        np.array(feat_inds),
        np.array(thresholds),
        np.array(polarities),
    )


def plot_boosting_examples():
    np.random.seed(0)

    mm = 150
    X = np.random.rand(mm, 2)
    thresh_pos = 0.6
    y = 2 * ((X[:, 0] < thresh_pos) & (X[:, 1] < thresh_pos)).astype(int) - 1

    for T in [1, 2, 4, 5, 10]:
        plt.figure()
        plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', linewidths=0.5, label='+1', color="blue")
        plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='x', linewidths=0.5, label='-1', color="red")

        alphas, feat_inds, thresholds, polarities = stump_booster(X, y, T)

        x1_coords = np.linspace(0, 1, 100)
        x2_coords = np.linspace(0, 1, 100)
        Z = np.zeros((100, 100))

        for ii, x1 in enumerate(x1_coords):
            for jj, x2 in enumerate(x2_coords):
                pred_sum = 0.0
                for alpha, j, s, d in zip(alphas, feat_inds, thresholds, polarities):
                    val = x1 if j == 0 else x2
                    h = d * (1 if (val - s) >= 0 else -1)
                    pred_sum += alpha * h
                Z[jj, ii] = np.sign(pred_sum)

        # draw decision boundary contour at 0
        plt.contour(x1_coords, x2_coords, Z, levels=[0], colors='k', linewidths=2)
        plt.title(f'Iterations = {T}')
        # plt.xlabel('x₁')
        # plt.ylabel('x₂')
        plt.gca().set_aspect('equal', 'box')
        # plt.legend()
        plt.show()

if __name__ == "__main__":
    plot_boosting_examples()
