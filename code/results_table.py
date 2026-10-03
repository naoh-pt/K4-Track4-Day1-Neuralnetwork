"""results_table.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
from pathlib import Path


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    exp_id = result["cfg"]["exp_id"]
    if not exp_id or Path(exp_id).name != exp_id or exp_id in {".", ".."}:
        raise ValueError("exp_id must be a plain filename")
    path = directory / f"{exp_id}.json"
    payload = {key: result[key] for key in ("cfg", "history", "summary")}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    return str(path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    directory = Path(results_dir)
    if not directory.exists():
        return []
    with_results = []
    for path in directory.glob("*.json"):
        with path.open(encoding="utf-8") as handle:
            with_results.append(json.load(handle))
    return sorted(with_results, key=lambda result: result["cfg"]["exp_id"])


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    row = {**result["cfg"], **result["summary"]}
    if "hidden" in row:
        row["hidden"] = "-".join(str(width) for width in row["hidden"])
    row["figure_file"] = f"figures/{row['exp_id']}.png"
    row["notes"] = notes
    if eval_scores is not None:
        accuracy = eval_scores.get("acc", eval_scores.get("accuracy",
                                                   eval_scores.get("eval_acc")))
        macro_f1 = eval_scores.get("macro_f1", eval_scores.get("eval_macro_f1"))
        if accuracy is None or macro_f1 is None:
            raise ValueError("eval_scores needs accuracy and macro_f1")
        row["eval_acc"] = accuracy
        row["eval_macro_f1"] = macro_f1
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước (openpyxl):
      1. wb = openpyxl.load_workbook(template_path)   # KHÔNG dùng data_only=True (sẽ mất công thức)
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức (step0_gap_vs_lnC, gap_val_minus_train,
         delta_val_f1_vs_base, beyond_noise)
      4. wb.save(out_path)
    Sau khi lưu, mở file bằng Excel/LibreOffice để các công thức tính lại.
    """
    from openpyxl import load_workbook

    workbook = load_workbook(template_path)
    sheet = workbook["Experiments"]
    headers = {cell.value: cell.column for cell in sheet[1] if cell.value}
    formula_columns = {"step0_gap_vs_lnC", "gap_val_minus_train",
                       "delta_val_f1_vs_base", "beyond_noise"}
    for row_number, values in enumerate(rows, start=2):
        for key, value in values.items():
            if key not in headers or key in formula_columns:
                continue
            cell = sheet.cell(row=row_number, column=headers[key])
            if cell.data_type != "f":
                cell.value = value
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
