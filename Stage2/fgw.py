import numpy as np
from parse_pdb import parse_pdb
from embed_esm2 import get_esm_embeddings, load_esm

# computes the pairwise distance matrix

def pairwise_dist(x):
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def squared_feature_dist(F1, F2):
    diff = F1[:, None, :] - F2[None, :, :]
    return np.sum(diff ** 2, axis=-1)


# computes the Sinkhorn transport plan

def sinkhorn(C, a, b, eps=0.05, n_iter=30):
    K = np.exp(-C / eps)

    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(n_iter):
        u = a / (K @ v + 1e-9)
        v = b / (K.T @ u + 1e-9)

    return np.diag(u) @ K @ np.diag(v)

# computes the FGW loss based off both the structure and the features

def fgw_loss(C1, C2, F1, F2, T, alpha=0.7):

    # structural term
    a = T @ np.ones(T.shape[1])
    b = T.T @ np.ones(T.shape[0])

    term_struct = (
        np.sum((C1 ** 2) * np.outer(a, a)) +
        np.sum((C2 ** 2) * np.outer(b, b)) -
        2 * np.trace(C1 @ T @ C2 @ T.T)
    )

    # feature term
    term_feat = np.sum(squared_feature_dist(F1, F2) * T)

    return alpha * term_struct + (1 - alpha) * term_feat


def fgw_cost_matrix(C1, C2, T):
    a = T.sum(axis=1)
    b = T.sum(axis=0)

    term1 = (C1 ** 2) @ a
    term2 = (C2 ** 2) @ b
    cross = C1 @ T @ C2.T

    return term1[:, None] + term2[None, :] - 2 * cross


def compute_fgw_from_features(
        X,
        Y,
        F1,
        F2,
        alpha=0.7,
        eps=0.05,
        sinkhorn_iter=30):
    C1 = pairwise_dist(X)
    C2 = pairwise_dist(Y)

    n, m = len(X), len(Y)
    a = np.ones(n) / n
    b = np.ones(m) / m

    T = np.outer(a, b)
    M_feat = squared_feature_dist(F1, F2)

    for _ in range(sinkhorn_iter):
        M_geom = fgw_cost_matrix(C1, C2, T)
        C = alpha * M_geom + (1 - alpha) * M_feat
        T = sinkhorn(C, a, b, eps=eps, n_iter=20)

    return fgw_loss(C1, C2, F1, F2, T, alpha)

# computes the FGW distance

def compute_fgw(pdb1, pdb2,
                alpha=0.7,
                eps=0.05,
                sinkhorn_iter=30,
                device="cpu",
                model=None,
                batch_converter=None):
    if model is None or batch_converter is None:
        model, _, batch_converter = load_esm()

    # load structures
    X, seq1 = parse_pdb(pdb1)
    Y, seq2 = parse_pdb(pdb2)

    # ESM-2 embeddings
    F1 = get_esm_embeddings(seq1, model, batch_converter, device=device)
    F2 = get_esm_embeddings(seq2, model, batch_converter, device=device)

    return compute_fgw_from_features(
        X,
        Y,
        F1,
        F2,
        alpha=alpha,
        eps=eps,
        sinkhorn_iter=sinkhorn_iter,
    )


if __name__ == "__main__":
    pdb1 = "proteinA.pdb"
    pdb2 = "proteinB.pdb"

    model, _, batch_converter = load_esm()
    score = compute_fgw(pdb1, pdb2, model=model, batch_converter=batch_converter)

    print("FGW distance:", score)