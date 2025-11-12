import numpy as np
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.optimize import linear_sum_assignment

def purity_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total = len(y_true)
    clusters = np.unique(y_pred)
    correct = 0
    for c in clusters:
        mask = (y_pred == c)
        true_labels = y_true[mask]
        if len(true_labels) == 0:
            continue
        counts = np.bincount(true_labels)
        correct += np.max(counts)
    return correct / total

def cluster_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / y_pred.size
    return acc

def evaluate(y_true, F):
    y_true = np.array(y_true)
    F = np.array(F)
    
    if F.ndim == 2 and F.shape[1] > 1:
        y_pred = np.argmax(F, axis=1)
    else:
        y_pred = F.flatten()
    
    acc = cluster_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    purity = purity_score(y_true, y_pred)
    
    metrics = {
        'ACC': acc,
        'NMI': nmi,
        'ARI': ari,
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'Purity': purity
    }
    return metrics

