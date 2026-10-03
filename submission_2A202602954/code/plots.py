"""plots.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg, history = result["cfg"], result["history"]
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    try:
        for metric, label in (("train_loss", "Train"), ("val_loss", "Validation")):
            axes[0].plot(epochs, history[metric], label=label)
        for metric, label in (("val_acc", "Accuracy"),
                              ("val_macro_f1", "Macro-F1")):
            axes[1].plot(epochs, history[metric], label=label)
        axes[2].plot(epochs, history["grad_norm"], label="Gradient norm")
        for axis, title, ylabel in zip(
            axes, ("Loss", "Validation metrics", "Gradient norm before clipping"),
            ("Loss", "Score", "Global L2 norm")
        ):
            axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
            axis.legend()
            best = result.get("summary", {}).get("best_epoch")
            if best is not None and best > 0:
                axis.axvline(best, color="gray", linestyle="--", alpha=0.5)
        fig.suptitle(
            f"{cfg['exp_id']} | {cfg['optimizer']} lr={cfg['lr']} "
            f"batch={cfg['batch']} hidden={cfg['hidden']} loss={cfg['loss']}"
        )
        fig.tight_layout()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        raise ValueError("At least one result is required")
    fig, axis = plt.subplots(figsize=(8, 5))
    try:
        for result in results:
            history = result["history"]
            if metric not in history:
                raise ValueError(f"Metric {metric!r} is missing from {result['cfg']['exp_id']}")
            axis.plot(history["epoch"], history[metric], label=result["cfg"]["exp_id"])
        axis.set(title=title or metric, xlabel="Epoch", ylabel=metric)
        axis.legend()
        fig.tight_layout()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
