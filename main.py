import numpy as np
from readdata import load_multiview_data
from initial import robust_initialize_with_mask
from admm import admm_iteration
from evaluate import evaluate

# ---------------------------- Parameter setting ------------------------- #
mat_file = "3Sources.mat"
missing_rate = 0.5
rank = 20           
k_nn = 3           
random_state = 42

# ---------------------------- Data loading ------------------------- #
X_views, y, cluster_number, M_mask = load_multiview_data(
    mat_file, missing_rate=missing_rate, seed=random_state
)
n_views = len(X_views)
n_samples = X_views[0].shape[0]

print(f"\n[Data Info]")
print(f"Views: {n_views}, Samples: {n_samples}, Clusters: {cluster_number}")
for v, Xv in enumerate(X_views):
    print(f" - View {v}: shape={Xv.shape}")

# ---------------------------- Initialize ------------------------- #
Xhat_views, Q_views, E_views, Z, F, alpha_v, barQ, S_views, Lz = \
    robust_initialize_with_mask(X_views, M_mask, cluster_number,
                                rank=rank, k_nn=k_nn, random_state=random_state)

# ---------------------------- ADMM ------------------------- #
W_list = []
for v in range(n_views):
    Wv = np.tile(M_mask[:, v][:, np.newaxis], (1, X_views[v].shape[1]))
    W_list.append(Wv)

Xhat_list, Q_list, E_list, S_list, Z, F, alpha_v, barQ, Lz, loss_list = admm_iteration(
    X_list=Xhat_views,  
    W_list=W_list,      
    Q_list=Q_views,
    E_list=E_views,
    S_list=S_views,
    Z=Z,
    F=F,
    alpha_v=alpha_v,
    lambda1=1.0,
    lambda2=0.1,
    lambda3=0.1,
    lambda4=0.1,
    lambda5=0.1,
    r=3,
    k_nn=k_nn,
    k=cluster_number,
    max_iter=50
)

# ---------------------------- Evaluate ------------------------- #
metrics = evaluate(y, F)
print(metrics)
