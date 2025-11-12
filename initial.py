import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csgraph

def robust_initialize_with_mask(X_views, M_mask, n_clusters, rank=50, k_nn=10, eps=1e-6, random_state=42):
    rng = np.random.default_rng(random_state)
    n_views = len(X_views)
    n_samples = X_views[0].shape[0]

    Xhat_views = []
    Q_views = []
    E_views = []
    S_views = []

    for v in range(n_views):
        X = X_views[v].copy()
        mask = M_mask[:, v].reshape(-1,1)

        X_fill = X.copy()
        col_mean = np.nanmean(X_fill, axis=0)
        inds = np.where(np.isnan(X_fill))
        X_fill[inds] = np.take(col_mean, inds[1])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_fill)

        U, S, VT = np.linalg.svd(X_scaled, full_matrices=False)
        r_v = min(rank, U.shape[1], VT.shape[0])
        U_r = U[:, :r_v]
        S_r = np.diag(S[:r_v])
        VT_r = VT[:r_v, :]

        Qv = U_r @ np.sqrt(S_r)  
        Qv = Qv / (np.linalg.norm(Qv, axis=0, keepdims=True) + eps)
        Q_views.append(Qv)

        Ev = VT_r.T @ np.sqrt(S_r)  
        E_views.append(Ev)

        Xhat_scaled = Qv @ Ev.T  
        Xhat = X.copy()
        Xhat_scaled = Xhat_scaled * scaler.scale_ + scaler.mean_  
        Xhat_missing = (mask == 0)
        Xhat_missing = np.repeat(Xhat_missing, X.shape[1], axis=1)
        Xhat[Xhat_missing] = Xhat_scaled[Xhat_missing]
        Xhat_views.append(Xhat)

    for v in range(n_views):
        X = Xhat_views[v]
        nbrs = NearestNeighbors(n_neighbors=min(k_nn, n_samples-1)).fit(X)
        distances, indices = nbrs.kneighbors(X)
        sigma2 = np.median(distances[:, 1:] ** 2)
        S = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j_idx, d in zip(indices[i], distances[i]):
                S[i, j_idx] = np.exp(-d**2 / (2*sigma2 + eps))
        S = (S + S.T) / 2
        S[S < 0] = 0
        S_views.append(S)

    Z = np.mean(S_views, axis=0)
    Z = (Z + Z.T)/2
    Z[Z < 0] = 0

    Lz = csgraph.laplacian(Z, normed=True)

    eigvals, eigvecs = np.linalg.eigh(Lz)
    idx = np.argsort(eigvals)[:n_clusters]
    F = eigvecs[:, idx]

    alpha_v = np.ones(n_views) / n_views

    barQ = np.mean(Q_views, axis=0)

    return Xhat_views, Q_views, E_views, Z, F, alpha_v, barQ, S_views, Lz

