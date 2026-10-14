Họ tên: Đặng Đỉnh Đoàn · Mã học viên: 2A202602927

Trạng thái: **đã chạy xong E0–E3 và E4 trên GPU (Kaggle 2×T4); số liệu và nhận xét bên dưới lấy từ lần chạy đó**.
Các khối giữa hai dòng `AUTO` do `python -m solution.report --write` điền từ `report/evidence/*.json`;
phần "Diễn giải" dưới mỗi khối là nhận xét viết tay, dựa trên `report/breakdown.json`
(`python -m solution.breakdown`), `report/analysis.json` và các ảnh trong `report/figures/`.
Một seed (42) cho mỗi setup; chạy trên Kaggle T4 nên thời gian/VRAM chỉ để so sánh tương đối.

<!-- AUTO:VERSION:START -->
`dataset_version`: `d10-21e4070301`. Số liệu lấy từ trọng tài, cùng split test.
<!-- AUTO:VERSION:END -->

## Bảng tổng hợp

<!-- AUTO:TABLE:START -->
| Mã | Tham số train | VRAM (GB) | Phút train | F1 | CI95 | FP/ảnh | pothole | crack | manhole |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | 0 | 4.309 | — | 0.0845 | [0.0623, 0.1093] | 1.5498 | 0.1176 | 0.0277 | 0.1369 |
| E1 | 33030144 | 6.496 | 43.868 | 0.2667 | [0.2330, 0.3024] | 1.4695 | 0.1835 | 0.1872 | 0.5489 |
| E2 | 142135296 | 8.189 | 42.426 | 0.3356 | [0.2966, 0.3752] | 1.2508 | 0.2022 | 0.2704 | 0.6137 |
| E3 | 148426752 | 8.369 | 58.596 | 0.4483 | [0.4089, 0.4879] | 0.9743 | 0.2963 | 0.3828 | 0.7099 |
| E4_DATA_010 | 148426752 | 8.133 | 58.246 | 0.2583 | [0.2262, 0.2909] | 1.8328 | 0.1512 | 0.1949 | 0.5238 |
| E4_DATA_100 | 148426752 | 8.369 | 58.596 | 0.4483 | [0.4089, 0.4879] | 0.9743 | 0.2963 | 0.3828 | 0.7099 |

E0 không train; VRAM E0 đo khi inference. Các run khác ghi peak allocated VRAM khi train.
<!-- AUTO:TABLE:END -->

E4 chọn **kích thước dữ liệu: 10% và 100%**. Setup nền được chọn bằng F1 validation,
giữ nguyên rank, alpha, learning rate, bước tối ưu, seed, prompt, toạ độ và effective batch.
Nhánh 100% dùng lại đúng lần chạy setup nền; chấm bằng `score.py` với tên E4 riêng.
Không chọn setup theo kết quả test.

## 10 câu hỏi

### Câu 1 — Kiến trúc và khối được cập nhật

ViT biến ảnh thành đặc trưng; projector ghép patch và chuyển sang chiều ẩn của LLM.
DeepStack đưa thêm đặc trưng trung gian của ViT vào các tầng LLM. LLM sinh JSON tự hồi quy.
E1 cập nhật LoRA trên attention/MLP của LLM; E2 thêm toàn bộ projector;
E3 thêm LoRA trên attention/MLP của ViT. Embedding, lm_head và trọng số nền của ViT/LLM được đóng băng.
Projector gồm `visual.merger` và `visual.deepstack_merger_list`, không chỉ merger cuối.
Code kiểm tra 252 module LLM, thêm 96 module ViT ở E3, ghi danh sách tham số trainable.
Đếm `quant_state.shape` cho trọng số 4-bit để không nhầm với kích thước bộ nhớ đóng gói.

<!-- AUTO:Q1:START -->
Đo trên model: ViT backbone 306,242,560; projector 109,105,152; LLM + embedding 4,022,468,096; tổng 4,437,815,808 tham số.

E1: 33,030,144 trainable; E2: 142,135,296 trainable; E3: 148,426,752 trainable.
<!-- AUTO:Q1:END -->

**Diễn giải.** Tỉ lệ tham số train trên tổng 4.438 tỉ: E1 33.0M (0.74%), E2 142.1M (3.20%), E3 148.4M (3.34%).
Phần tăng từ E1 lên E2 gần như toàn bộ là projector (109.1M, chiếm 77% số tham số train của E2),
còn LoRA trên ViT ở E3 chỉ thêm 6.3M (0.14% tổng) nhưng cho mức tăng F1 lớn nhất (xem Câu 7).


### Câu 2 — E0 và lỗi model gốc

`parse_valid_rate` dùng parser của đề trên output thô. Parser có thể chấp nhận markdown fence
hoặc trích JSON từ văn bản dư, nên hợp lệ theo parser không đồng nghĩa tuân thủ hoàn toàn prompt.
Nhóm lỗi trong `report/analysis.json` có thể chồng lấp: JSON hỏng, thừa/sai vị trí box,
thiếu/sai vị trí box, nhầm lớp có IoU đủ cao. Không kết luận hai lỗi phổ biến trước khi chạy.

<!-- AUTO:Q2:START -->
E0 parse_valid_rate=0.8585. Nhóm lỗi nhiều nhất theo quy tắc đếm ảnh: `missed_or_mislocalized_gt` (282 ảnh), `extra_or_mislocalized_prediction` (247 ảnh). Ví dụ output thô `vlcsnap-2025-02-19-17h03m26s014`:

````text
[{"label":"pothole","bbox_2d": [472, 657, 500, 683]},{"label":"crack","bbox_2d": [0, 683, 1000, 750]},{"label":"manhole","bbox_2d": [575, 457, 583, 467]},{"label":"manhole","bbox_2d": [565, 457, 573, 467]}]
````
<!-- AUTO:Q2:END -->

**Diễn giải (từ `work/e0.jsonl`).** 44/311 output của E0 không parse được: 38 trong số đó bị cắt ở 256 token vì model
sa vào vòng lặp xếp liên tiếp các box `crack` kề nhau (ví dụ `[393,420,450,450],[450,420,500,450],[500,420,550,450],...`),
6 ảnh còn lại hỏng cú pháp (ví dụ thừa một `]` sau bbox: `"bbox_2d": [520,329,530,352]],`). Không có markdown fence
và không có nhãn ngoài ba lớp. Về nội dung, E0 thiên lệch mạnh về `manhole`: dự đoán 290 box `manhole` trong khi test chỉ có 163,
và chỉ 32 box `pothole` trong khi test có 121 (tính trên các output parse được); vì vậy precision manhole 0.107 và recall pothole 0.074.
Hai nhóm lỗi theo quy tắc đếm ảnh chồng lấp nhau (282 ảnh bỏ sót/sai vị trí, 247 ảnh có box thừa/sai vị trí) nên không cộng được.
Sau finetune chế độ lặp vẫn còn nhưng hiếm: E1 có 2 output bị cắt (parse_valid 0.994), E3 có 1 (0.997).


### Câu 3 — Thang 0–1000 và kiểm chứng sai thang

YOLO dùng tâm/kích thước chuẩn hoá; đổi sang góc bằng `(xc±w/2, yc±h/2)*1000`,
làm tròn và kẹp biên. Vẽ lại: `x_pixel=x_norm*width/1000`, `y_pixel=y_norm*height/1000`.
Nếu hiểu box norm1000 thành pixel, trục x bị nhân thêm 1000/640 và y thêm 1000/360;
box lệch vị trí và kích thước, IoU/F1 thường giảm; không khẳng định luôn bằng 0.
Kiểm chứng dùng cùng output E1 và GT, chỉ đổi `coord` khi evaluate; đây là kiểm tra
thang chấm, không phải thí nghiệm thay prompt/nhãn trong quá trình train.

<!-- AUTO:Q3:START -->
Cùng output E1: đúng norm1000 F1=0.2667; hiểu nhầm thành pixel F1=0.0000. Không train thêm trong kiểm tra này.
<!-- AUTO:Q3:END -->

**Diễn giải.** Khi hiểu nhầm toạ độ chuẩn hoá 0–1000 thành pixel, F1 của cùng output E1 rơi từ 0.2667 xuống 0.0000:
x bị nhân thêm 1000/640 = 1.56 và y thêm 1000/360 = 2.78 nên hầu hết box lệch khỏi vị trí đúng, IoU không đạt ngưỡng 0.5.
Con số 0.0000 cho thấy lỗi thang đo là lỗi hệ thống, không phải nhiễu; vì vậy mô hình, prompt và bộ chấm phải dùng chung một thang.


### Câu 4 — Mask loss

Mục tiêu là học `P(JSON | ảnh, prompt)`: copy input_ids vào labels, gán -100 cho prompt.
Token ảnh, system/user và assistant header đều bị mask; padding cũng -100.
Chỉ JSON và `<|im_end|>` được giám sát. Model tự dịch labels khi tính causal loss, không dịch lần nữa.
Code xác nhận prompt là tiền tố của chuỗi đầy đủ, từ chối cắt token ảnh để vừa max_length.
Loss toàn chuỗi làm model tốn năng lực tái tạo hướng dẫn/token đặc biệt;
loss có thể giảm mà grounding chưa cải thiện. Token ảnh vẫn tham gia forward và truyền gradient
về projector/LoRA ViT khi được mở, dù labels tại token ảnh bằng -100.

<!-- AUTO:Q4:START -->
Mẫu `vlcsnap-2025-02-26-20h36m10s957`: 220 token ảnh, 352 token bị mask và 57 token được giám sát.

````text
[{"label":"crack","bbox_2d":[583,583,895,897]},{"label":"crack","bbox_2d":[358,586,548,825]}]<|im_end|>
````
<!-- AUTO:Q4:END -->

**Diễn giải.** Trong mẫu kiểm tra có 409 token, 352 token bị mask (gồm 220 token ảnh) và chỉ 57 token (14%) được giám sát:
JSON và `<|im_end|>`. Loss chỉ tính trên 57 token này nên giá trị loss thấp không đo trực tiếp chất lượng grounding:
loss E4 10% xuống 0.003 mà F1 test chỉ 0.258 (Câu 10) là ví dụ.


### Câu 5 — Lượng tử hoá và float32 của projector

Trọng số 4-bit đóng gói không dùng như tham số đầy đủ cho AdamW.
LoRA học ma trận phụ cạnh trọng số lượng tử, không thay thế việc mở toàn bộ projector.
Nếu lượng tử hoá cả ViT bao gồm merger, E2/E3 không cập nhật đầy đủ merger theo cách của đề.
Về nguyên lý có thể lượng tử backbone và loại trừ riêng projector; trong lab,
cách dùng trong code là `llm_int8_skip_modules=["model.visual","lm_head"]` và kiểm tra module thật.
Đường dẫn đầy đủ `model.visual` cần thiết với matcher của Transformers 5.19.0;
chỉ `visual` không khớp các module `model.visual.blocks.*`.
Projector E2/E3 chuyển float32 trước khi tạo optimizer để giữ trọng số/gradient ổn định,
phù hợp với unscale của fp16 GradScaler; autocast vẫn có thể tính phép toán ở fp16.
Float32 không phải yêu cầu tuyệt đối cho mọi phương pháp train: đây là lựa chọn của vòng lặp fp16 trong lab.
Lưu adapter LoRA và projector riêng; suy luận nạp cả hai, kiểm tra tên tensor khớp.

### Câu 6 — Split theo phiên quay

Frame cùng video có thể gần trùng nhau; chia ngẫu nhiên theo ảnh đưa thông tin cảnh
vào cả train và test, khiến điểm phản ánh ghi nhớ cảnh nhiều hơn khả năng chuyển sang phiên mới.
Giữ nguyên prepare_data.py; kiểm tra không trùng session/image_id giữa split,
đối chiếu số đếm với meta.json. Split theo phiên không bảo đảm cân bằng lớp.

<!-- AUTO:Q6:START -->
- train: 1422 ảnh, 13 session; pothole=1074 box, crack=1750 box, manhole=691 box
- val: 276 ảnh, 2 session; pothole=66 box, crack=470 box, manhole=103 box
- test: 311 ảnh, 3 session; pothole=121 box, crack=299 box, manhole=163 box

Không có session trùng giữa các split theo audit.
<!-- AUTO:Q6:END -->

**Diễn giải.** Phân bố theo session rất lệch: train có 13 session, val chỉ 2 (276 ảnh, 470 box crack nhưng chỉ 66 pothole),
test 3 session (213 / 83 / 15 ảnh). Do đó F1 val và F1 test khác nhau đáng kể (E3: 0.338 trên val, 0.448 trên test;
E1: 0.198 và 0.267) — khác biệt này nhiều khả năng do session và phân bố lớp khác nhau (audit không thấy rò rỉ session). Thứ tự E1 < E2 < E3 giống nhau ở cả val và test,
nên việc chọn E3 bằng val không bị đảo ngược trên test.


### Câu 7 — E1/E2/E3 có đáng chi phí không?

So sánh chất lượng cùng tài nguyên, không mặc định mở nhiều khối luôn tốt hơn.
E2 thêm gradient/trạng thái AdamW cho projector khoảng 109M tham số;
E3 thêm gradient qua ViT, nên có thể tăng VRAM/thời gian. Kết luận dựa trên số đo.
Hai điểm F1 và việc hai CI riêng chồng lấp chưa đủ cho kiểm định;
bổ sung bootstrap theo cặp trên cùng ảnh để ước lượng CI95 của hiệu F1.

<!-- AUTO:Q7:START -->
`e2-e1`: ΔF1=0.0689, paired CI95=[0.0332, 0.1064]; CI loại trừ 0 trong bootstrap theo ảnh.

`e3-e2`: ΔF1=0.1127, paired CI95=[0.0744, 0.1514]; CI loại trừ 0 trong bootstrap theo ảnh.

Setup nền chọn bằng validation: E3 (F1 val=0.3378). Chi phí ghi ở bảng trên.
<!-- AUTO:Q7:END -->

**Diễn giải.**

| Bước | ΔF1 test (paired CI95) | Tham số train thêm | VRAM thêm | Thời gian thêm |
|---|---|---:|---:|---:|
| E1 → E2 (mở projector) | +0.069 [0.033, 0.106] | +109.1M | +1.69 GiB | −1.4 phút (nhiễu) |
| E2 → E3 (LoRA ViT) | +0.113 [0.074, 0.151] | +6.3M | +0.18 GiB | +16.2 phút (+38%) |

Chi phí bộ nhớ của E2 khớp tính toán: 109.1M × (4 B trọng số + 4 B gradient + 8 B trạng thái AdamW) ≈ 1.63 GiB, đo được +1.69 GiB (cột VRAM tính theo GiB).
Theo chi phí-lợi ích, E3 là hợp lý nhất: chỉ thêm 6.3M tham số (+0.18 GiB) nhưng tăng F1 nhiều hơn bước mở projector.
Điều này gợi ý lớp đặc trưng thị giác gốc (ViT) chưa phù hợp với vết nứt mảnh/ổ gà nhỏ trong ảnh dashcam, và chỉ chỉnh
phần ghép sang LLM (projector) là chưa đủ. Đây là suy luận từ một seed, chưa có thí nghiệm loại trừ (ví dụ chỉ LoRA ViT không mở projector).
Thời gian E3 dài hơn do phải lan truyền gradient qua ViT; thời gian giữa các run không được kiểm soát hoàn toàn
(E1/E2 chạy song song với suy luận E0 trên GPU còn lại, E3 chạy một mình) nên chỉ nên đọc theo bậc độ lớn.
E3 ghi nhận 3 bước tràn gradient fp16 bị `GradScaler` bỏ qua (không tính vào 300 bước); E1/E2 không có bước nào.


### Câu 8 — Test nhỏ và khoảng tin cậy

Trọng tài bootstrap theo ảnh 1.000 lần; đọc độ rộng CI và paired CI của hiệu F1.
CI hiệu chứa 0: chưa đủ bằng chứng thay đổi có hướng, không chứng minh hai model tương đương.
CI không bao gồm biến thiên do seed train. Frame cùng session còn tương quan,
nên bootstrap theo ảnh có thể lạc quan; kết luận cho phiên mới cần thêm session độc lập.
Giữ nguyên CI trọng tài và ghi rõ giới hạn này.

<!-- AUTO:Q8:START -->
Test thực tế 311 ảnh. Độ rộng CI95: E0=0.0469; E1=0.0694; E2=0.0786; E3=0.0790.
<!-- AUTO:Q8:END -->

**Diễn giải.** 311 ảnh nhưng chỉ 3 session (213 / 83 / 15 ảnh), nên số mẫu độc lập thực chất nhỏ hơn 311 nhiều:
các frame cùng video gần như trùng cảnh, bootstrap theo ảnh vì thế cho CI hẹp hơn mức bất định thật khi chuyển sang phiên mới.
Bằng chứng trong dữ liệu: F1 test của cùng một model thay đổi theo session (E3: 0.565 trên session `num-4` 15 ảnh, 0.447 trên `vlc-2025-02-18` 213 ảnh, 0.436 trên `vlc-2025-02-19` 83 ảnh),
và khoảng cách val–test của E3 là 0.11, lớn hơn nửa độ rộng CI95 (0.04). Mặt khác, thứ tự E1 < E2 < E3 đúng ở **cả ba** session
(theo thứ tự session như trên — E1: 0.356 / 0.286 / 0.214; E2: 0.478 / 0.331 / 0.327; E3: 0.565 / 0.447 / 0.436), nên kết luận về hướng của khác biệt đáng tin hơn
độ lớn chính xác của nó. CI không bao gồm biến thiên do seed: mỗi setup chỉ train một lần (seed 42), nên khác biệt
nhỏ như E2 so với E1 (+0.069) cần thận trọng hơn khác biệt lớn như E3 so với E2 (+0.113) hoặc 100% so với 10% dữ liệu (+0.190).
Kết luận "E3 tốt hơn E1" khá chắc; xếp hạng hai setup gần nhau sẽ cần nhiều seed và thêm session độc lập.


### Câu 9 — Phân tích lỗi theo lớp

Xem F1 từng lớp, số mẫu train/val/test, diện tích box và ảnh có FP/FN.
Box nhỏ dễ mất chi tiết qua resize/patch merging; crack mảnh có nhiều cách đặt box;
manhole và pothole có thể giống nhau. Đây là giả thuyết cần kiểm tra ảnh, chưa phải nguyên nhân đã chứng minh.
Ảnh minh hoạ chọn từ cả nhóm đúng và lỗi: GT xanh, dự đoán đỏ và TP/FP/FN.

<!-- AUTO:Q9:START -->
Trong E1, lớp kém nhất là pothole: F1=0.1835; diện tích box trung vị trên test=0.4193% ảnh. Có 6 ảnh trong [bộ ảnh minh hoạ](report/figures/predictions/README.md). Cần đọc từng ảnh để xác nhận giả thuyết về nguyên nhân lỗi.
<!-- AUTO:Q9:END -->

**Diễn giải (từ 6 ảnh E1 trong `report/figures/predictions/` và số đo E3 trong `report/breakdown.json`).**

*Ba ảnh đúng* (`vlcsnap-00202`, `-00204`, `-00206`) đều là một nắp cống `manhole` đơn lẻ, vùng tối hình bầu dục nằm ngay trước capô
trên cùng một đoạn đường; box đỏ nhỏ hơn hoặc lệch nhẹ so với box xanh nhưng IoU vẫn ≥ 0.5. Cả ba là các frame liên tiếp
của một lần quay nên chỉ là một loại tình huống dễ, không đại diện cho cả test.

*Ba ảnh lỗi* cho ba kiểu khác nhau:
- `vlcsnap-2025-02-19-17h03m02s980` (TP=0, FP=6, FN=6): đường núi có cụm nứt/ổ gà dày; model xác định đúng *vùng* nhưng
  chia thành 6 box `crack` nhỏ, trong khi nhãn là các box chồng lấn kích thước khác — lỗi định vị/độ chi tiết, không phải bỏ sót vùng.
- `vlcsnap-2025-02-19-17h08m59s876` (TP=0, FP=7, FN=5): nhãn là các vết nứt ngang dài và hai ổ gà; model xếp nhiều box hẹp
  thẳng đứng chồng lên nhau ở giữa ảnh. Đây là kiểu "xếp ô" đã thấy trong output E0 bị cắt (Câu 2).
- `vlcsnap-2025-02-18-23h15m48s025` (TP=0, FP=9, FN=1): chỉ có một box nhãn nhỏ sát lề phải; model đặt một hàng 7 box `crack`
  dọc một đường ngang bên trái (giống vạch kẻ/mép đường) cộng 2 box bên phải. Lỗi dương tính giả kiểu nhầm vạch kẻ là giả thuyết,
  tôi chưa kiểm chứng bằng thí nghiệm.

*Số đo của E3 (model tốt nhất)*:
- F1 theo lớp: manhole 0.710 (P 0.800, R 0.638) > crack 0.383 > pothole 0.296; lớp kém nhất vẫn là pothole, trong khi
  diện tích box trung vị trên test: pothole 0.42%, manhole 0.67%, crack 1.98% diện tích ảnh.
- Recall theo kích thước box nhãn: box nhỏ (<0.2% ảnh) 4/37 = 10.8%; trung bình (0.2–1%) 100/241 = 41.5%; lớn (≥1%) 152/305 = 49.8%.
  Ngay cả box lớn mới tìm được một nửa, nên nguyên nhân không chỉ là độ phân giải/box nhỏ mà còn là độ mơ hồ của ranh giới
  vết nứt kéo dài (annotator chia box khác nhau).
- 303 box dự đoán sai (FP) của E3 chia thành: 146 không chạm GT cùng lớp, 120 gần đúng nhưng IoU 0.1–0.5 (lỗi định vị), 37 nhầm lớp
  (IoU ≥ 0.5 với GT khác lớp). Nhầm lớp chỉ chiếm ~12% nên lỗi chủ yếu là *ở đâu và bao nhiêu box*, không phải *loại gì*.
- Trung bình 1.80 box dự đoán/ảnh so với 1.88 box nhãn/ảnh: model không tràn box. Số ảnh có ≥ 6 box dự đoán giảm từ 11 (E1)
  xuống 3 (E3), phù hợp với việc kiểu "xếp ô" giảm khi mở ViT.

Giới hạn: ảnh minh hoạ do `solution/analysis.py` chọn từ E1, chưa vẽ lại cho E3 nên phần nhận xét ảnh là của E1 và các
số E3 chỉ nói về thống kê, không khẳng định E3 sửa đúng những ảnh này.


### Câu 10 — E4 data-size ablation

Chọn setup có F1 validation cao nhất ở E1–E3. Tập 10% là prefix danh sách train shuffle seed 42;
tập 100% chứa toàn bộ danh sách. Giữ 300 bước tối ưu và effective batch 8.
Tập nhỏ được lặp lại nhiều hơn: so lượng dữ liệu ở cùng ngân sách tối ưu, không cùng số epoch.
Không thay rank, prompt hoặc thang toạ độ. Nhánh 100% dùng lại baseline và raw predictions;
hai file evidence sinh qua trọng tài. Nhìn F1, CI95 và paired CI hiệu trước khi kết luận.

<!-- AUTO:Q10:START -->
Setup nền E3; tập 10%=142 ảnh, 100%=1422 ảnh. `e4_data_100-e4_data_010`: ΔF1=0.1900, paired CI95=[0.1511, 0.2289]; CI loại trừ 0 trong bootstrap theo ảnh.
<!-- AUTO:Q10:END -->

**Diễn giải.** Cùng setup E3 (cùng 300 bước, effective batch 8, cùng seed), chỉ đổi dữ liệu: F1 test 0.258 [0.226, 0.291]
với 10% (142 ảnh) so với 0.448 [0.409, 0.488] với 100% (1422 ảnh); ΔF1 = +0.190, paired CI95 [0.151, 0.229] loại trừ 0.
Hai CI riêng không chồng lấp. Precision giảm từ 0.458 xuống 0.231 trong khi recall chỉ từ 0.439 xuống 0.293,
FP/ảnh tăng 0.97 → 1.83: tập nhỏ chủ yếu làm tăng box sai, phù hợp với học thuộc.
Loss train cuối của nhánh 10% là 0.003 (nhánh 100%: 0.564) vì 300 bước × 8 ảnh = 2400 mẫu là khoảng 17 lượt qua 142 ảnh
nhưng chỉ khoảng 1.7 lượt qua 1422 ảnh. Vì vậy so sánh này gộp hai yếu tố: *ít dữ liệu* và *lặp lại nhiều*; kết quả không tách
được hai yếu tố, và dừng sớm có thể cứu một phần nhánh 10%. Đáng chú ý: E3 với 10% dữ liệu (0.258) gần bằng E1 với 100% dữ liệu
(0.267), tức mở ViT/projector không bù được việc thiếu dữ liệu. Nhánh 100% dùng lại dự đoán của E3 nên hai số này
trùng E3 theo thiết kế, không phải một lần lặp độc lập; tập 10% là một mẫu cố định (seed 42), chưa kiểm tra mẫu 10% khác.


## E5 — Bonus đã chuẩn bị

Reward Python thuần có test JSON hỏng, box hoàn hảo, sai lớp, thừa/trùng box, rỗng, box khổng lồ.
Công thức `0.2+0.8*sum_IoU/max(n_gt,n_pred,1)` cho JSON hợp lệ;
hai tập rỗng được 1, JSON hỏng được 0. Ghép cùng lớp, IoU ≥0.5, mỗi box dùng một lần.
GRPO khởi tạo từ SFT chọn trên val, 4 generations, 128 completion tokens, 60 bước,
lr 5e-6, beta=0, use_vllm=False.
Sau khi chạy kiểm tra tỷ lệ `[]`, box lớn, reward và F1 để phát hiện reward hacking.

<!-- AUTO:E5:START -->
Chưa chạy GRPO, chưa có bonus evidence hoặc kết luận cải thiện.
<!-- AUTO:E5:END -->

## Nhìn lại

Ưu tiên kiểm tra nhãn/toạ độ, prefix mask và vòng đời projector trước khi tăng thời gian train.
E1−E0 F1=0.1821 (≥ 0.10) và E1 parse_valid_rate=0.9936 (≥ 0.95): hai ngưỡng của rubric đều đạt.
Kết quả chính: E0 0.085 → E1 0.267 → E2 0.336 → E3 0.448; cải thiện lớn nhất đến từ việc chỉnh phía thị giác (ViT),
và dữ liệu (E4) ảnh hưởng nhiều hơn việc mở thêm khối. Hạn chế đã biết: một seed mỗi setup, test chỉ 3 session,
ảnh minh hoạ chỉ cho E1, và 3 bước tràn gradient fp16 bị bỏ qua ở E3/E4.
