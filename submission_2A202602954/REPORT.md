# Báo cáo Lab Day 1 — Phan Trọng Hoàn — 2A202602954

## 1. Thiết lập

Thực nghiệm đã lưu chạy trên Google Colab, GPU Tesla T4, PyTorch 2.11.0+cu130 (output trong `code/lab.ipynb`). Dữ liệu Forest CoverType có 54 đặc trưng và 7 lớp. `split_metadata.csv` xác định sẵn 464.809 mẫu train và 116.203 mẫu eval; từ train tách 20% validation theo nhãn với seed 42, được 371.847 train và 92.962 validation. Chỉ 10 cột số đầu được chuẩn hóa bằng trung bình và độ lệch chuẩn của phần train sau tách; 44 cột nhị phân giữ nguyên. Trung bình và độ lệch chuẩn đo trên 10 cột train đã chuẩn hóa lần lượt xấp xỉ 0 và 1. Nhãn eval không tham gia chọn cấu hình, epoch hay chuẩn hóa.

M-base là `54 → 256 → 128 → 7`, ReLU ở các lớp ẩn, đầu ra là logits, 47.879 tham số. Baseline chính `base-s1`: He, CE, SGD momentum 0,9, LR 0,1, batch 512, 20 epoch, seed 1, FP32, dropout 0, không clip. Accuracy đoán lớp đa số trên validation là 0,4876. Mọi so sánh dưới đây dùng validation; con số của từng lần chạy nằm ở dòng cùng `exp_id` trong `experiments.xlsx` và bản JSON gốc ở repo làm việc.

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả đã ghi |
|---|---:|
| Tham số / shape logits | 47.879 / `(8, 7)`; logits hữu hạn |
| Gradient sau một backward | Cả 6 tensor tham số có gradient khác `None`, chuẩn L2 dương |
| CE bước 0, seed 1 / `ln(7)` | 2,269062 / 1,945910 |
| Quá khớp 20 mẫu | Bước 19: CE = 7,07969×10⁻⁵, accuracy = 1,000 |
| Baseline seed 1 / 2: val macro-F1 | 0,861318 / 0,851844 |
| Baseline: val accuracy, trung bình ± độ lệch chuẩn mẫu | 0,909640 ± 0,000989 |
| Baseline: val macro-F1, trung bình ± độ lệch chuẩn mẫu | 0,856581 ± 0,006699 |

Loss bước 0 của seed 1 hữu hạn nhưng cao hơn `ln(7)` 0,323152; đó không phải một điều kiện phải khớp chính xác khi khởi tạo ngẫu nhiên. Seed 2 có bước 0 là 1,978157. Phép thử 20 mẫu cho thấy pipeline có thể học đến 100% trên một lô nhỏ. Với `base-s1`, train loss đo ở eval mode giảm 0,4761 → 0,2030 và val loss giảm 0,4771 → 0,2273 từ epoch 1 đến 20; best epoch là 20, val accuracy 0,910340. [Đường baseline](figures/base-s1.png).

Ngưỡng tham khảo `2σ` của val macro-F1 từ **hai** seed là 0,013398. Đây chỉ là ước lượng thô, không phải kiểm định chắc chắn. Dự đoán đã được người thực hiện xác nhận: LR quá nhỏ hội tụ chậm, LR quá lớn có thể kém ổn định; kỳ vọng LR 0,1 phù hợp cho baseline. Ba LR pilot chạy cùng 5 epoch: `lr-sgdm-0.01` đạt macro-F1 0,626110; `lr-sgdm-0.03` đạt 0,694520; `lr-sgdm-0.1` đạt 0,772516. LR 0,1 cũng có val loss pilot thấp nhất (0,340008), nên được chọn cho baseline 20 epoch bằng validation. [Ảnh pilot LR 0,1](figures/lr-sgdm-0.1.png).

## 3. Kết quả theo chủ đề

Các dự đoán dưới đây do người thực hiện cung cấp trong cuộc trao đổi sau khi chạy; notebook gốc không lưu dấu thời điểm viết chúng. Báo cáo tách dự đoán được cung cấp khỏi kết quả đo, không tự khẳng định đã ghi chúng trong notebook trước lần chạy. Các chênh lệch macro-F1 dưới 0,013398 được xem là chưa đủ rõ so với nhiễu seed đã đo.

### 3.1 Hàm mất mát — CE vs MSE

Dự đoán được cung cấp: CE sẽ tốt hơn MSE cho phân loại nhiều lớp.

`loss-ce` đạt val macro-F1 0,861318 và accuracy 0,910340; `loss-mse` (logits so với one-hot, trung bình trên mẫu và 7 lớp) đạt 0,732234 và 0,871991. Chênh macro-F1 0,129084 vượt xa ngưỡng nhiễu. CE cho tín hiệu gradient trực tiếp hơn với phân loại khi dự đoán sai nặng; MSE trên logits/one-hot có mục tiêu và thang đo khác. **Không so trực tiếp trị số CE loss với MSE loss.** Xem [CE](figures/loss-ce.png) và [MSE](figures/loss-mse.png).

### 3.2 Bộ tối ưu hóa

Dự đoán được cung cấp: Adam/AdamW có thể hội tụ nhanh và đạt macro-F1 cao hơn SGD, còn SGD momentum có thể ổn định hơn.

Mỗi bộ tối ưu được thử ba LR; bảng là ứng viên có val macro-F1 cao nhất **trong các LR đã thử**.

| Bộ tối ưu | `exp_id` | LR | Val macro-F1 | Best epoch |
|---|---|---:|---:|---:|
| SGD momentum | `opt-sgd_momentum-0.1` | 0,1 | 0,861318 | 20 |
| Adam | `opt-adam-0.003` | 0,003 | 0,876029 | 20 |
| AdamW | `opt-adamw-0.003` | 0,003 | 0,876029 | 20 |

Trong dải LR đã thử, Adam 0,003 hơn baseline seed 1 là 0,014711 macro-F1, nhỉnh hơn `2σ` 0,013398; kết luận còn yếu vì chỉ có hai seed baseline và một seed Adam. `opt-adam-0.0003`/`0.001` lần lượt đạt 0,796835/0,844677, cho thấy kết luận phụ thuộc LR. Adam và AdamW cho **cùng** val macro-F1 ở từng LR tương ứng vì mọi run này đặt `weight_decay=0`; thí nghiệm này không đo lợi ích của weight decay tách riêng. [Ảnh so sánh optimizer](figures/compare_optimizer.png).

### 3.3 Hyper-parameter — batch size

Dự đoán được cung cấp: batch quá lớn có thể giảm khả năng tổng quát hóa; batch 512 có thể cân bằng tốc độ và hiệu năng.

Giữ LR 0,1 và các yếu tố khác như baseline: `batch-128` đạt macro-F1 0,858914, 5,113 giây/epoch; `batch-512` đạt 0,861318, 1,295 giây/epoch; `batch-2048` đạt 0,811601, 0,341 giây/epoch. Batch 128 và 512 chênh 0,002404, nhỏ hơn `2σ`, còn batch 512 nhanh hơn khoảng 3,95 lần. Batch 2048 nhanh hơn mỗi epoch nhưng macro-F1 thấp hơn rõ; cùng 20 epoch, batch lớn có ít bước cập nhật hơn nên không thể diễn giải như chỉ thay chi phí tính toán. Xem [batch 128](figures/batch-128.png), [512](figures/batch-512.png), [2048](figures/batch-2048.png).

### 3.4 Dropout

Dự đoán được cung cấp: dropout nhỏ có thể giảm quá khớp, nhưng dropout cao có thể gây thiếu khớp.

`drop-0` đạt macro-F1 0,861318; `drop-0.3` đạt 0,775098; `drop-0.5` đạt 0,675185. Cả hai mức dropout làm giảm chất lượng validation trong cấu hình này. Ở baseline epoch 20, train loss 0,2030 và val loss 0,2273 có khoảng cách nhỏ; dropout mạnh có thể cản học nhiều hơn lợi ích giảm quá khớp. Train loss trong hình được đo ở eval mode để so sánh cùng thang với validation. Xem [dropout 0,3](figures/drop-0.3.png) và [0,5](figures/drop-0.5.png).

### 3.5 Gradient clipping

Dự đoán được cung cấp: clipping có thể hữu ích khi gradient bất ổn hoặc LR cao.

Grad norm trung bình trước clip của baseline theo epoch ở khoảng 0,53–0,60. Với LR cao 1,0, `clip-highlr-none` đạt macro-F1 0,597744; cùng LR nhưng clip ở 0,5, `clip-highlr-0.5` đạt 0,809314, tăng 0,211570. Clip giúp **cứu chất lượng học** ở LR cao bằng cách giới hạn bước cập nhật khi gradient lớn. Cả hai run đều ghi `diverged=False`, nên không có bằng chứng rằng clip đã ngăn một lỗi NaN/Inf. Kết quả clip vẫn thấp hơn baseline LR 0,1 (0,861318), vì clipping không biến LR quá cao thành cấu hình tối ưu. Xem [không clip](figures/clip-highlr-none.png) và [clip 0,5](figures/clip-highlr-0.5.png).

### 3.6 Mixed precision

Dự đoán được cung cấp: mixed precision có thể giảm thời gian và bộ nhớ nhưng cũng có rủi ro bất ổn số.

`amp-fp32`: macro-F1 0,861318, 1,318 giây/epoch; `amp-fp16`: 0,824633 tại best epoch 8, 1,752 giây/epoch, `diverged=True`; `amp-bf16`: 0,854725, 1,533 giây/epoch, không diverge. BF16 gần FP32 trong phạm vi `2σ` nhưng chậm hơn; FP16 vừa chậm hơn vừa không ổn định trong setup T4/PyTorch này. Peak memory được log khoảng 164,25 MB cho cả ba; thước đo này không cho thấy lợi ích bộ nhớ đáng kể. Mixed precision ở mạng nhỏ có thể chịu overhead autocast/scaler lớn hơn phần tiết kiệm tính toán; đây là giải thích cơ chế, không phải phép đo riêng overhead. Xem [FP32](figures/amp-fp32.png), [FP16](figures/amp-fp16.png), [BF16](figures/amp-bf16.png).

### 3.7 Khởi tạo tham số

Dự đoán được cung cấp: He phù hợp nhất với ReLU; Xavier vẫn có thể hoạt động tốt, còn zeros có thể thất bại do đối xứng.

`init-he` đạt macro-F1 0,861318; `init-xavier` 0,858233, chênh 0,003085 dưới `2σ`. Trên cùng batch validation trước train, độ lệch chuẩn kích hoạt sau ba lớp Linear của He là `[0,666, 0,646, 0,593]`; Xavier `[0,278, 0,220, 0,197]`. He giữ thang kích hoạt ổn định hơn qua các lớp ReLU, dù macro-F1 cuối khá gần. `init-normal` và `init-default` đạt 0,840995 và 0,844361. `init-zeros` chỉ đạt 0,093650 macro-F1 và accuracy 0,487597, gần chiến lược đoán lớp đa số; kích hoạt ban đầu cả ba lớp đều có std 0. Trọng số bằng nhau ở các nơ-ron ẩn gây đối xứng nên chúng không học các đặc trưng khác nhau. Xem [He](figures/init-he.png), [Xavier](figures/init-xavier.png), [zeros](figures/init-zeros.png).

## 4. Đánh giá cuối trên tập eval

Chọn `opt-adam-0.003` **trước khi xem eval**, dựa trên val macro-F1 0,876029 và val loss 0,209431, đồng hạng tốt nhất với AdamW trong các cấu hình đã thử và không diverge. Cấu hình cuối: Adam, LR 0,003, CE, batch 512, M-base `[256,128]`, dropout 0, He, không clip, FP32, seed 1, 20 epoch. `best_epoch=20`. Tập eval chỉ được dùng để tạo và chấm `predictions_eval.csv` sau lựa chọn.

| Cấu hình | Seed | `n_eval` | Val macro-F1 | Eval macro-F1 | Eval accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline `base-s1` | 1 | 116.203 | 0,861318 | 0.8646731514537692 | 0.9088319578668365 |
| Final `opt-adam-0.003` | 1 | 116.203 | 0,876029 | **0.8793854637439035** | **0.9157767011178714** |

Điểm baseline lấy từ `baseline_eval_result.json`, điểm final lấy từ `eval_result.json`; cả hai chấm trên 116.203 mẫu. Final tăng **0,0147123122901343 macro-F1** (khoảng 0,014712) và **0,0069447432510349 accuracy** (khoảng 0,006945) so với baseline trên eval. Đây là so sánh của **một seed cho mỗi cấu hình**, không phải ước lượng độ bất định của chênh lệch eval. Eval chỉ được dùng **sau khi** chọn cấu hình bằng validation; không dùng để chỉnh LR, chọn epoch hay tuning mô hình.

### 4.1 Phân tích lỗi theo lớp

| Lớp | Support | F1 baseline | Precision final | Recall final | F1 final |
|---:|---:|---:|---:|---:|---:|
| 0 | 42.368 | 0,903968 | 0,914970 | 0,906698 | 0,910815 |
| 1 | 56.661 | 0,922619 | 0,922402 | 0,933146 | 0,927743 |
| 2 | 7.151 | 0,901961 | 0,910356 | 0,914557 | 0,912452 |
| 3 | 549 | 0,810036 | 0,870000 | 0,792350 | 0,829361 |
| 4 | 1.899 | 0,766883 | 0,831555 | 0,779884 | **0,804891** |
| 5 | 3.473 | 0,825749 | 0,853579 | 0,827527 | 0,840351 |
| 6 | 4.102 | 0,921497 | 0,934317 | 0,925890 | 0,930084 |

Final có F1 cao hơn baseline ở cả 7 lớp trong hai file scorer, đặc biệt lớp 4 tăng khoảng 0,038008 và lớp 5 tăng khoảng 0,014602. Lớp 4 vẫn có F1 thấp nhất; 352/1.899 mẫu lớp 4 bị dự đoán thành lớp 1, nhiều nhất trong các nhầm lẫn của lớp này. Lớp 3 có support thấp nhất (549), recall 0,792350 và 79 mẫu nhầm thành lớp 2. Ma trận nhầm lẫn đầy đủ nằm trong `eval_result.json`; hai cặp nhầm lớn khác là 0→1 (3.674) và 1→0 (3.270). Mất cân bằng lớp là một lý do khả dĩ; không có phép thử nào ở đây xác nhận nguyên nhân đặc trưng. Nếu làm tiếp, cần kiểm tra phân bố đặc trưng và ví dụ sai theo lớp bằng train/validation trước khi thay mô hình.

## 5. Trả lời câu hỏi dẫn dắt

1. Trong ba LR mỗi bộ tối ưu, Adam và AdamW cùng cao nhất ở 0,003 (0,876029), SGD momentum cao nhất ở 0,1 (0,861318). So tại cùng LR tùy tiện sẽ lẫn ảnh hưởng LR với bộ tối ưu; weight decay bằng 0 nên AdamW chưa được thử đúng khác biệt chính của nó.
2. Dropout 0,3/0,5 không giúp ở đây: macro-F1 giảm xuống 0,775098/0,675185. Chỉ nên kỳ vọng nó hữu ích khi khoảng cách train–val cho thấy quá khớp và xác nhận lại trên validation.
3. Clipping giới hạn chuẩn gradient toàn cục trước bước cập nhật. Cặp LR 1,0 tăng macro-F1 0,597744 → 0,809314 khi clip 0,5, nhưng không vượt baseline LR 0,1.
4. Trên T4 với mạng này, FP16 chậm hơn FP32 (1,752 so với 1,318 giây/epoch) và run diverge; BF16 1,533 giây/epoch. Không có tốc độ tăng trong dữ liệu đã đo.
5. Zeros tạo các nơ-ron ẩn đối xứng, không học biểu diễn khác nhau; macro-F1 chỉ 0,093650. He được thiết kế cho ReLU nên giữ phương sai kích hoạt qua lớp tốt hơn Xavier ở phép đo bước 0, dù hai macro-F1 cuối chênh dưới nhiễu seed.
6. Nếu loss không giảm sau 2.000 bước, trước tiên kiểm tra **dữ liệu/nhãn và logits–loss** (shape, nhãn `0..6`, CE nhận logits thô); tiếp theo thử **quá khớp 20 mẫu và kiểm tra gradient** để xác nhận có cập nhật; sau đó xem **LR, grad norm và NaN/Inf** để nhận ra bước quá nhỏ, quá lớn hoặc bất ổn. Các phép thử bước 0, 20 mẫu và clipping trong báo cáo là bằng chứng trực tiếp cho thứ tự chẩn đoán này.

## 6. Hạn chế và điều bất ngờ

Chỉ có hai seed baseline và một seed cho mỗi cấu hình khác; `2σ` vì vậy rất thiếu chắc chắn. Mỗi cấu hình mới chỉ có một điểm eval, nên chưa đo được nhiễu seed của mức cải thiện eval. Pilot LR có 5 epoch, không so ngang với run 20 epoch. Batch khác nhau có số bước cập nhật khác nhau. FP16 bị đánh dấu diverge nên số epoch hoàn thành và thời gian/epoch không đại diện cho run đủ 20 epoch. Notebook gốc không lưu các dự đoán trước chạy; dự đoán được người thực hiện cung cấp sau và không thể kiểm chứng thời điểm viết. Kết quả chỉ phản ánh dải cấu hình đã thử trên T4, chưa chứng minh ưu thế tổng quát của Adam hay tác dụng của AdamW khi weight decay dương.

## 7. Phụ lục

Nguồn số: `results/*.json` ở repo làm việc (32 lần chạy), `experiments.xlsx`, output thật trong `code/lab.ipynb`, `baseline_eval_result.json` cho baseline và `eval_result.json` cho final. Hai file bằng chứng baseline (`baseline_eval_result.json`, `baseline_predictions_eval.csv`) chỉ ở repo root, không nằm trong submission. Artifact nộp: `REPORT.md`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, `figures/`, `code/`. Thư mục `results/` là khuyến nghị, không bắt buộc theo README và không được copy theo yêu cầu chỉ lấy file cần nộp. Báo cáo này không dùng checkpoint và không chạy lại phép chấm eval.
