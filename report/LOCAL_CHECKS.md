# Kiểm tra local — 14/10/2026

Trạng thái tại thời điểm kiểm tra: code và cấu hình hoàn thiện, **chưa chạy model, inference hoặc training GPU**.
(Cập nhật sau đó: E0–E4 đã chạy trên Kaggle 2×T4, xem `REPORT.md` và `report/evidence/`.)
Không tải trọng số Qwen, không tạo prediction/evidence thực nghiệm giả.

## Đã kiểm tra

```bash
PYTHONPATH=/tmp/day10-cpu-check python -m pytest -q tests solution/tests
```

Kết quả: **98 passed, 1 skipped**. Torch 2.6.0+cpu được cài riêng trong `/tmp` để kiểm tra tensor;
không thay torch hệ thống hoặc dùng CUDA. Test skipped là test dữ liệu thật của đề,
vì môi trường chưa có dataset và biến D10_RAW.

Các kiểm tra bao gồm chuyển toạ độ, thang pixel/norm1000, mask answer/im_end,
padding đa phương thức, từ chối prefix sai/truncation, sampler resume,
hai config E4 chỉ thay train_fraction, chọn module LoRA và reward IoU.
Pipeline CPU dùng **fixture tổng hợp trong thư mục tạm của pytest** để kiểm tra
trọng tài → phân tích → sáu ảnh minh hoạ → render report; fixture không phải kết quả lab.
Notebook được xác nhận hợp lệ, mọi cell compile được, không có output/execution count,
RUN_GPU/RUN_E5 đều False. CLI thiếu --allow-gpu dừng trước khi nạp thư viện model.

`python -m compileall -q solution` và `git diff --check` đã qua.
`git diff -- scripts tests configs` rỗng: không sửa công cụ/test/config gốc.
Các wheel của năm phiên bản thư viện ghim trong đề tải được vào `/tmp` để đối chiếu API;
chỉ tải gói Python, không tải/training model.

## Việc cần kết quả cloud

`python scripts/check_submission.py` hiện có **7 FAIL, 1 WARN**:

- Thiếu evidence E0, E1, E2, E3 và E4: 5 FAIL.
- REPORT còn marker số liệu chưa đo: 1 FAIL.
- Git mới có 1/5 commit: 1 FAIL.
- Chưa có data/splits/meta.json để đối chiếu: 1 WARN.

Đây là trạng thái trước khi bạn chạy GPU, không phải kết luận bài đã đạt rubric.
CUDA/NF4, vòng train trên Qwen thật, VRAM, thời gian, chất lượng E0–E4 và GRPO E5
cần được kiểm chứng ở Colab/Kaggle. Hướng dẫn chạy và xử lý lỗi nằm trong
[HUONG_DAN_CHAY.md](../HUONG_DAN_CHAY.md).

## Cập nhật sau lỗi dữ liệu trên Colab

Xử lý nhãn width/height bằng 0 và box biến mất sau làm tròn: bỏ khỏi đáp án train,
ghi ví dụ/số lượng trong audit, giữ nguyên ground truth/split/số đếm của đề.
Test hồi quy xác nhận không tạo hình học giả hoặc làm mất nhãn gốc.
Kiểm tra lại: **100 passed, 1 skipped**, không chạy GPU.

## Cập nhật skip modules sau lỗi E0 trên Colab

Đổi `visual` thành đường dẫn đầy đủ `model.visual` trong cấu hình bitsandbytes.
Đã kiểm chứng bằng chính hàm `should_convert_module` trong wheel Transformers 5.19.0:
backbone ViT, merger, DeepStack merger và lm_head được loại trừ; projection LLM vẫn được lượng tử.
Kiểm tra lại **100 passed, 1 skipped**. Chưa thực hiện nạp model hoặc chạy GPU ở local.
