# %% 
import numpy as np
import matplotlib.pyplot as plt


def compute_margins(X, y, alphas, feat_inds, thresholds, polarities=None):
    X = np.asarray(X)
    y = np.asarray(y)
    alphas = np.asarray(alphas)
    feat_inds = np.asarray(feat_inds, dtype=int)
    thresholds = np.asarray(thresholds)

    if polarities is None:
        polarities = np.ones_like(alphas)
    else:
        polarities = np.asarray(polarities)

    F = np.zeros(X.shape[0])
    for alpha, j, s, d in zip(alphas, feat_inds, thresholds, polarities):
        h = np.sign(X[:, j] - s)
        h[h == 0] = 1
        F += alpha * (d * h)

    alpha_norm = np.sum(np.abs(alphas))

    margins = (y * F) / alpha_norm

    return margins.tolist()


def stump_booster(X, y, T, track_margins=False):
    """
    AdaBoost implementation for tree stumps WLA. 
    """
    m, n = X.shape
    w = np.ones(m) / m

    alphas = []
    feat_inds = []
    thresholds = []
    polarities = []

    margins = [] # T x |X| matrix

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

        if track_margins: 
            margins_t = compute_margins(X, y, alphas, feat_inds, thresholds, polarities)
            margins.append(margins_t)


        # update example weights
        w *= np.exp(-alpha * y * best_pred)
        w /= np.sum(w)

    return (
        np.array(alphas),
        np.array(feat_inds),
        np.array(thresholds),
        np.array(polarities),
        np.array(margins)
    )

def generate_data(mm): 
    np.random.seed(0)
    X = np.random.rand(mm, 2)
    thresh_pos = 0.6
    y = 2 * ((X[:, 0] < thresh_pos) & (X[:, 1] < thresh_pos)).astype(int) - 1
    return X, y 

def plot_decision_boundary(X, y, T, X_test=None, y_test=None):
    """
    Train AdaBoost (stump_booster) for T iterations on (X, y) and plot the 
    resulting decision boundary. If X_test, y_test are provided, they're shown
    in the same class colors but with lower alpha to appear faint.
    """
    
    plt.scatter(
        X[y ==  1, 0], X[y ==  1, 1],
        marker='o', label='+1 (train)',
        color='blue', edgecolor='k', lw=0.5
    )
    plt.scatter(
        X[y == -1, 0], X[y == -1, 1],
        marker='x', label='-1 (train)',
        color='red', edgecolor='k', lw=0.5
    )
    

    if X_test is not None and y_test is not None:
        plt.scatter(
            X_test[y_test ==  1, 0], X_test[y_test ==  1, 1],
            marker='o', label='+1 (test)',
            color='blue', alpha=0.1, edgecolor='k', lw=0.5
        )
        plt.scatter(
            X_test[y_test == -1, 0], X_test[y_test == -1, 1],
            marker='x', label='-1 (test)',
            color='red', alpha=0.1, edgecolor='k', lw=0.5
        )
    

    alphas, feat_inds, thresholds, polarities, _ = stump_booster(X, y, T)
    
    x1_min, x1_max = 0, 1
    x2_min, x2_max = 0, 1
    xx = np.linspace(x1_min, x1_max, 200)
    yy = np.linspace(x2_min, x2_max, 200)
    X1, X2 = np.meshgrid(xx, yy)
    
    Z = np.zeros_like(X1)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            vote = 0.0
            for alpha, feat, θ, d in zip(alphas, feat_inds, thresholds, polarities):
                val = X1[i, j] if feat == 0 else X2[i, j]
                h = d * (1 if (val - θ) >= 0 else -1)
                vote += alpha * h
            Z[i, j] = np.sign(vote)
    
    plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
    
    plt.title(f'AdaBoost Decision Boundary ({T} iterations)')
    # plt.legend(loc='best', fontsize='small')

X, y = generate_data(150)
X_test, y_test = generate_data(5000)
plot_decision_boundary(X, y, 10, X_test, y_test)

alphas, feat_inds, thresholds, polarities, _ = stump_booster(X, y, T=10) 
margins = compute_margins(X, y, alphas, feat_inds, thresholds, polarities)

GAMMA_STAR = np.percentile(margins, 0.10) # unregularized
print(GAMMA_STAR)

# %% 
def plot_boosting_examples():
    np.random.seed(0)

    X, y = generate_data(150)
    X_test, y_test = generate_data(1000)

    for T in [1, 2, 4, 5, 10]:
        plt.figure()
        plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', linewidths=0.5, label='+1', color="blue")
        plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='x', linewidths=0.5, label='-1', color="red")
        plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], marker='o', linewidths=0.5, label='1', color="lightskyblue")
        plt.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], marker='x', linewidths=0.5, label='-1', color="lightcoral")


        alphas, feat_inds, thresholds, polarities, _ = stump_booster(X, y, T)

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
