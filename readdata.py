import scipy.io
import numpy as np

def load_multiview_data(mat_path, missing_rate=0.0, seed=42):
    mat = scipy.io.loadmat(mat_path)
    X_cell = mat['X']
    y = mat['y'].squeeze()

    X_views = [X_cell[i, 0] for i in range(X_cell.shape[0])]
    n_views = len(X_views)
    n_samples = X_views[0].shape[0]

    rng = np.random.default_rng(seed)
    M_mask = np.ones((n_samples, n_views), dtype=np.uint8)  

    missing_per_view = int(n_samples * missing_rate)

    for v in range(n_views):
        if v == n_views - 1:
            candidates = np.where(np.sum(M_mask[:, :v], axis=1) > 0)[0]
        else:
            candidates = np.arange(n_samples)

        if len(candidates) < missing_per_view:
            raise ValueError(f"Missing rate too high.")

        missing_idx = rng.choice(candidates, missing_per_view, replace=False)
        M_mask[missing_idx, v] = 0

    cluster_number = len(np.unique(y))
    return X_views, y, cluster_number, M_mask
