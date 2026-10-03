"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import csv
import copy
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm)
    if cm.shape != (7, 7):
        raise ValueError("Confusion matrix must have shape (7, 7)")
    tp = np.diag(cm).astype(np.float64)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(7, dtype=np.float64),
                   where=denominator != 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    parts = [model(X[start:start + batch_size]).argmax(dim=1).to(torch.int64)
             for start in range(0, len(X), batch_size)]
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.int64,
                                                     device=X.device)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    if batch_size <= 0 or len(X) == 0 or len(X) != len(y):
        raise ValueError("Evaluation needs nonempty matching X/y and positive batch_size")
    if loss_name not in {"ce", "mse"}:
        raise ValueError(f"Unsupported loss {loss_name!r}")
    model.eval()
    loss_sum = 0.0
    correct = 0
    cm = torch.zeros((7, 7), dtype=torch.int64, device=y.device)
    for xb, yb in iterate_batches(X, y, batch_size, shuffle=False):
        logits = model(xb)
        # compute_loss returns a sample mean; weight each batch by its size.
        loss_sum += compute_loss(logits, yb, loss_name).item() * len(yb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        cm += torch.bincount(yb * 7 + pred, minlength=49).reshape(7, 7)
    return {"loss": loss_sum / len(y), "acc": correct / len(y),
            "macro_f1": macro_f1_from_confusion(cm.cpu().numpy())}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        # Mean squared error over classes, then over samples.
        target = F.one_hot(y, num_classes=7).to(logits.dtype)
        return F.mse_loss(logits, target)
    raise ValueError(f"Unsupported loss {loss_name!r}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    missing = set(DEFAULT_CFG) - set(cfg)
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    if cfg["lr"] is None:
        raise ValueError("cfg['lr'] must be chosen using validation before training")
    if cfg["loss"] not in {"ce", "mse"}:
        raise ValueError(f"Unsupported loss {cfg['loss']!r}")
    if cfg["batch"] <= 0 or cfg["epochs"] <= 0:
        raise ValueError("batch and epochs must be positive")
    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    if not len(X_tr) or not len(X_val):
        raise ValueError("Training and validation tensors must be nonempty")
    if any(t.device != X_tr.device for t in (y_tr, X_val, y_val)):
        raise ValueError("Training and validation tensors must share a device")
    device = X_tr.device
    precision = cfg["precision"]
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision {precision!r}")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training requires CUDA")
    if precision == "bf16" and not (device.type == "cpu" or
                                    device.type == "cuda" and torch.cuda.is_bf16_supported()):
        raise ValueError(f"BF16 autocast is unsupported on {device}")

    set_seed(cfg["seed"])
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"])
    if hidden in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[hidden]
    model = model.to(device)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"],
                                weight_decay=cfg["weight_decay"],
                                momentum=cfg["momentum"],
                                betas=cfg.get("betas", (0.9, 0.999)),
                                eps=cfg.get("eps", 1e-8))
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def autocast_context():
        if precision == "fp32":
            return nullcontext()
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type=device.type, dtype=dtype)

    # The first 50,000 training samples form a fixed, comparable subset each epoch.
    X_report, y_report = X_tr[:50_000], y_tr[:50_000]
    step0 = evaluate(model, X_val, y_val, cfg["loss"])
    history = {key: [] for key in ("epoch", "train_loss", "val_loss", "val_acc",
                                    "val_macro_f1", "grad_norm", "epoch_time_s")}
    diverged = not np.isfinite(step0["loss"])
    best_loss = step0["loss"] if not diverged else None
    best_epoch = 0
    best_acc = step0["acc"]
    best_f1 = step0["macro_f1"]
    best_state = {key: value.detach().cpu().clone()
                  for key, value in model.state_dict().items()}
    generator = torch.Generator(device="cpu").manual_seed(cfg["seed"])

    for epoch in range(1, cfg["epochs"] + 1):
        if diverged:
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model.train()
        norm_sum = 0.0
        steps = 0
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=generator):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                loss = compute_loss(model(xb), yb, cfg["loss"])
            if not torch.isfinite(loss).item():
                diverged = True
                break
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # Norm and clipping need unscaled gradients.
            grad_norm = clip_gradients(model.parameters(), cfg["clip_norm"])
            if not np.isfinite(grad_norm):
                diverged = True
                break
            norm_sum += grad_norm
            steps += 1
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
        if diverged:
            break
        train_metrics = evaluate(model, X_report, y_report, cfg["loss"])
        val_metrics = evaluate(model, X_val, y_val, cfg["loss"])
        if not np.isfinite(train_metrics["loss"]) or not np.isfinite(val_metrics["loss"]):
            diverged = True
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        for key, value in (("epoch", epoch), ("train_loss", train_metrics["loss"]),
                           ("val_loss", val_metrics["loss"]),
                           ("val_acc", val_metrics["acc"]),
                           ("val_macro_f1", val_metrics["macro_f1"]),
                           ("grad_norm", norm_sum / steps),
                           ("epoch_time_s", elapsed)):
            history[key].append(value)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_acc = val_metrics["acc"]
            best_f1 = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}

    summary = {
        "step0_loss": step0["loss"] if np.isfinite(step0["loss"]) else None,
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1] if history["epoch"] else None,
        "final_val_loss": (history["val_loss"][-1] if history["epoch"] else best_loss),
        "val_acc": best_acc, "val_macro_f1": best_f1,
        "time_per_epoch_s": (sum(history["epoch_time_s"]) / len(history["epoch"])
                             if history["epoch"] else 0.0),
        "peak_mem_MB": (torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda" else 0.0),
        "diverged": diverged,
    }
    return {"cfg": copy.deepcopy(cfg), "history": history, "summary": summary,
            "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    ids = np.asarray(row_id)
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    labels = np.asarray(preds)
    if ids.ndim != 1 or labels.ndim != 1 or len(ids) != len(labels):
        raise ValueError("row_id and preds must be matching one-dimensional arrays")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("row_id values must be unique")
    if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels > 6)):
        raise ValueError("preds must contain integer labels in 0..6")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "pred"))
        writer.writerows((row_id, int(pred)) for row_id, pred in zip(ids, labels))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"])
    model.load_state_dict(result["best_state"])
    model.to(data["X_eval"].device)
    predictions = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], predictions, pred_path)
