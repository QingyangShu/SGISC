import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.linalg import eigh

def preprocess_W_list(W_list, X_list):
    W_processed = []
    for v, Xv in enumerate(X_list):
        Wv = np.array(W_list[v])
        if Wv.ndim == 1 or Wv.shape[1] == 1:
            Wv = np.tile(Wv[:, np.newaxis], (1, Xv.shape[1]))
        W_processed.append(Wv)
    return W_processed

def update_Xhat(Xv, Qv, Ev, Wv):
    Xhat = Wv * Xv + (1 - Wv) * (Qv @ Ev.T)
    return Xhat

def update_Ev(Xhat, Qv, epsilon=1e-8):
    QtQ = Qv.T @ Qv
    Ev = np.linalg.solve(QtQ + epsilon * np.eye(QtQ.shape[0]), Qv.T @ Xhat).T
    return Ev

def update_Qv(Xhat, Ev, barQ, lambda1, lambda2):
    ETE = Ev.T @ Ev
    numerator = Xhat @ Ev + (lambda2 / lambda1) * barQ
    denominator = ETE + (lambda2 / lambda1) * np.eye(ETE.shape[0])
    Qv = numerator @ np.linalg.inv(denominator)
    return Qv

def construct_Sv(Xhat, k_nn=10):
    dist = pairwise_distances(Xhat)
    sigma = np.median(dist)
    if sigma == 0:
        sigma = 1.0
    K = np.exp(-dist**2 / (2 * sigma**2))
    Sv = np.zeros_like(K)
    for i in range(K.shape[0]):
        knn_idx = np.argsort(K[i, :])[-k_nn:]
        Sv[i, knn_idx] = K[i, knn_idx]
    Sv = np.maximum((Sv + Sv.T) / 2, 0)
    np.fill_diagonal(Sv, 0)
    return Sv

def update_Z(S_list, alpha_v, lambda5, r):
    V = len(S_list)
    Z_num = np.zeros_like(S_list[0])
    Z_den = np.zeros_like(S_list[0])
    for v in range(V):
        Z_num += (alpha_v[v] ** r) * S_list[v]
        Z_den += (alpha_v[v] ** r)
    Z = Z_num / (Z_den + lambda5)
    Z = (Z + Z.T) / 2
    return Z

def update_alpha(S_list, Z, Lz, lambda3, r):
    V = len(S_list)
    h2 = np.zeros(V)
    for v in range(V):
        diff = (Z - S_list[v])
        h2[v] = np.linalg.norm(diff, 'fro')**2 + lambda3 * np.trace(S_list[v].T @ Lz @ S_list[v])
    alpha_v = (1 / h2) ** (1 / (r - 1))
    alpha_v /= np.sum(alpha_v)
    return alpha_v

def update_F(Lz, k):
    _, vecs = eigh(Lz)
    F = vecs[:, :k]
    return F

def update_barQ(Q_list):
    return sum(Q_list) / len(Q_list)

def update_Lz(Z):
    D = np.diag(np.sum(Z, axis=1))
    Lz = D - Z
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.sum(Z, axis=1), 1e-12)))
    Lz_norm = D_inv_sqrt @ Lz @ D_inv_sqrt
    return Lz_norm

def compute_loss(X_list, Xhat_list, Q_list, E_list, S_list, Z, F, alpha_v, W_list,
                 lambda1=1.0, lambda2=0.1, lambda3=0.1, lambda4=0.1, lambda5=0.01, r=2):
    V = len(X_list)
    loss = 0.0
    barQ = update_barQ(Q_list)
    Lz = update_Lz(Z)
    for v in range(V):
        recon_loss = np.linalg.norm(W_list[v] * (Xhat_list[v] - X_list[v]), 'fro')**2
        lowrank_loss = np.linalg.norm(Xhat_list[v] - Q_list[v] @ E_list[v].T, 'fro')**2
        mmd_loss = lambda2 * np.linalg.norm(Q_list[v] - barQ, 'fro')**2
        graph_loss = alpha_v[v]**r * (np.linalg.norm((Z - S_list[v]), 'fro')**2 +
                                       lambda3 * np.trace(S_list[v].T @ Lz @ S_list[v]))
        loss += lambda1 * (recon_loss + lowrank_loss) + mmd_loss + graph_loss
    loss += lambda4 * np.trace(F.T @ Lz @ F)
    loss += lambda5 * np.linalg.norm(Z, 'fro')**2
    return loss

def admm_iteration(X_list, W_list, Q_list, E_list, S_list, Z, F, alpha_v,
                   lambda1=1.0, lambda2=0.1, lambda3=0.1, lambda4=0.1, lambda5=0.01, r=2, k_nn=10, k=5, max_iter=10):
    V = len(X_list)
    W_list = preprocess_W_list(W_list, X_list)
    loss_list = []

    for it in range(max_iter):
        Xhat_list = [update_Xhat(X_list[v], Q_list[v], E_list[v], W_list[v]) for v in range(V)]

        E_list = [update_Ev(Xhat_list[v], Q_list[v]) for v in range(V)]

        barQ = update_barQ(Q_list)
        Q_list = [update_Qv(Xhat_list[v], E_list[v], barQ, lambda1, lambda2) for v in range(V)]

        S_list = [construct_Sv(Xhat_list[v], k_nn) for v in range(V)]

        Z = update_Z(S_list, alpha_v, lambda5, r)

        Lz = update_Lz(Z)

        F = update_F(Lz, k)

        alpha_v = update_alpha(S_list, Z, Lz, lambda3, r)

        barQ = update_barQ(Q_list)
        
        loss = compute_loss(X_list, Xhat_list, Q_list, E_list, S_list, Z, F, alpha_v, W_list,
                            lambda1, lambda2, lambda3, lambda4, lambda5, r)
        loss_list.append(loss)
        print(f"Iter {it+1}/{max_iter}, Loss: {loss:.6f}")

    return Xhat_list, Q_list, E_list, S_list, Z, F, alpha_v, barQ, Lz, loss_list
