"""data.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    paths = [Path(processed_dir) / name for name in ("train.npz", "eval.npz")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing processed dataset: {path}. Run: python scripts/split_data.py"
            )
    splits = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            required = ("X", "y", "row_id") if path.name == "eval.npz" else ("X", "y")
            missing = set(required) - set(archive.files)
            if missing:
                raise ValueError(f"{path}: missing fields {sorted(missing)}")
            X, y = archive["X"], archive["y"]
            if X.dtype != np.float32 or X.ndim != 2 or X.shape[1] != 54:
                raise ValueError(f"{path}: X must be float32 with shape (N, 54)")
            if y.dtype != np.int64 or y.ndim != 1 or len(X) != len(y):
                raise ValueError(f"{path}: y must be int64 with shape (N,) matching X")
            if len(y) == 0 or np.any((y < 0) | (y > 6)):
                raise ValueError(f"{path}: labels must be nonempty and in 0..6")
            splits.extend((X, y))
            if "row_id" in required:
                row_id = archive["row_id"]
                if row_id.ndim != 1 or len(row_id) != len(X):
                    raise ValueError(f"{path}: row_id must have shape (N,) matching X")
                splits.append(row_id)
    return tuple(splits)


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    numeric = X_tr[:, :N_NUMERIC]
    if len(numeric) == 0:
        raise ValueError("Cannot fit a standardizer on an empty training split")
    mean = numeric.mean(axis=0, dtype=np.float64)
    std = numeric.std(axis=0, dtype=np.float64)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    result = X.copy()
    safe_std = np.where(std == 0, 1.0, std)
    result[:, :N_NUMERIC] = (result[:, :N_NUMERIC] - mean) / safe_std
    return result


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)
    data = {"eval_row_id": eval_row_id, "mean": mean, "std": std}
    for name, X, y in (("tr", X_tr, y_tr), ("val", X_val, y_val),
                       ("eval", X_eval, y_eval)):
        data[f"X_{name}"] = torch.as_tensor(
            apply_standardizer(X, mean, std), dtype=torch.float32, device=device
        )
        data[f"y_{name}"] = torch.as_tensor(y, dtype=torch.int64, device=device)
    majority_class = np.bincount(y_tr, minlength=7).argmax()
    print(f"train: {X_tr.shape}, val: {X_val.shape}, eval: {X_eval.shape}")
    print(f"Validation majority-class accuracy: {np.mean(y_val == majority_class):.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(X) != len(y) or X.device != y.device:
        raise ValueError("X and y must have matching sample counts and devices")
    if shuffle:
        permutation_device = generator.device if generator is not None else X.device
        indices = torch.randperm(len(X), generator=generator, device=permutation_device)
        indices = indices.to(X.device)
    else:
        indices = torch.arange(len(X), device=X.device)
    for start in range(0, len(X), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield X[batch_indices], y[batch_indices]
