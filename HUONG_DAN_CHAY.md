# Hướng dẫn chạy bài lab Day 10

Bài đã có code cho E0–E3, E4 thay lượng dữ liệu và E5 bonus. E0–E4 đã chạy trên Kaggle 2×T4 (kết quả ở `REPORT.md`); E5 chưa chạy.
Bạn chạy trên Colab/Kaggle theo thứ tự dưới đây. Không cần tự viết thêm vòng train.
Các file `scripts/`, `tests/`, `configs/e0–e5.yaml` của đề được giữ nguyên.

## 1. Mang code hiện tại lên cloud

Trên máy local, ở thư mục repo:

```bash
python -m solution.package source --out artifacts/day10_colab.zip
```

Mở `lab_day10.ipynb` trong Colab (Upload notebook) hoặc Kaggle (Import notebook).
Notebook đã điền code, có đủ bước 0–10 và mặc định `RUN_GPU = False`, `RUN_E5 = False`.
ZIP source dùng bản code hiện tại trên máy, tránh clone nhầm bản GitHub chưa được push.
ZIP không chứa dữ liệu ảnh, secret, trọng số hoặc `.git`.

Nếu muốn kiểm tra code trên CPU local trong môi trường riêng:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q tests solution/tests
```

Torch CPU chỉ dùng kiểm tra tensor, không nạp model hoặc train. Nếu bỏ torch,
test tensor được skip; trên Colab/Kaggle dùng torch có sẵn, không cài wheel CPU.

**Colab:** chọn runtime GPU T4; chạy cell bootstrap và upload `day10_colab.zip` khi được yêu cầu.
Cell bootstrap giải nén vào `/content/day10` và đổi working directory.
**Kaggle:** tạo notebook GPU, bật Internet, thêm ZIP code thành một input; đặt `CODE_ZIP`
trong cell bootstrap đến đường dẫn thật, ví dụ `/kaggle/input/day10-code/day10_colab.zip`.
Working directory sẽ là `/kaggle/working/day10`. Chỉ code/data output nằm ở working; input chỉ đọc.

Nếu muốn dùng Git thay ZIP, hãy commit/push code mới trước rồi clone repo;
không cần công khai dữ liệu hoặc trọng số.

## 2. Cài dependencies, lưu checkpoint

Chạy cell cài `requirements-colab.txt`, sau đó `pip check`. Giữ torch/torchvision do runtime cung cấp.
Nếu đã import các thư viện trước khi cài, restart session/kernel rồi chạy lại bootstrap.
Kiểm tra GPU bằng `!nvidia-smi -L`; đây chỉ là kiểm tra phần cứng.

Trên Colab, cell Drive mount tạo `/content/drive/MyDrive/day10_backups`.
Mỗi 100 bước optimizer, code lưu adapter, projector E2/E3, optimizer/scheduler/scaler/RNG
và sao lưu checkpoint/log lên Drive. Chạy từng setup trong process riêng để giải phóng VRAM.
Trên Kaggle, dùng Save Version hoặc tải output trước khi session kết thúc.
ZIP kết quả cuối **không chứa trọng số**; muốn tiếp tục train phải giữ riêng checkpoint đầy đủ.

Nếu không cần Drive, đặt `MOUNT_DRIVE=False`. Đổi `RUN_GPU` ở cell bootstrap sẽ reset
các biến cell đó, nên sau khi chạy lại bootstrap hãy chạy lại cell Drive để đặt BACKUP_DIR.

## 3. Chuẩn bị dữ liệu (CPU)

Notebook có ba cách; chọn một:

- `DATA_MODE='kaggle'`: Colab Secrets tên `KAGGLE_USERNAME`, `KAGGLE_KEY`, bật quyền truy cập notebook.
  Code đưa secret vào env, không in khoá. Kaggle cũng hỗ trợ Secrets nếu muốn dùng CLI.
- `DATA_MODE='source'`: Kaggle Add Input dataset của đề, đặt DATA_SOURCE đến thư mục chứa dataset.
  `prepare_data.py` tự tìm `images/` và `labels-YOLO/` trong thư mục con; không cần API key.
- `DATA_MODE='zip'`: tải/upload ZIP dataset, đặt DATA_ZIP đến đường dẫn ZIP. Không dùng ZIP source ở bước này.

Dataset: [road damage của Lorenzo Arcioni](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes).

Các lệnh tương đương:

```bash
python scripts/prepare_data.py --kaggle
# Hoặc: python scripts/prepare_data.py --source /kaggle/input/ten-input-thuc-te
# Hoặc: python scripts/prepare_data.py --zip /content/dataset.zip
python -m solution.run audit
python -m solution.analysis --labels-only
python -m pytest -q tests solution/tests
```

Xem `report/data_audit.json`, dataset_version, số ảnh/box và không có session trùng split.
Dữ liệu của đề dự kiến 1.422 train / 276 val / 311 test; code đọc và kiểm tra số thực tế.
Sáu ảnh nhãn ở `report/figures/labels/`. Mẫu JSON ở bước 2 notebook phải khớp vị trí box.

Nếu nhãn gốc có width/height bằng 0 hoặc box biến mất sau làm tròn/kẹp biên,
code bỏ box đó khi tạo đáp án train và ghi `unusable_target_boxes` cùng ví dụ vào audit.
Giữ nguyên ảnh, split, số box gốc và ground truth của trọng tài; không tăng kích thước box giả
để vượt kiểm tra. Nhãn lỗi vẫn có thể bị tính FN khi chấm, cần ghi giới hạn dữ liệu trong báo cáo.

## 4. Bật GPU và chạy E0

Sau khi chuẩn bị dữ liệu, đặt `RUN_GPU=True` trong bootstrap, chạy lại bootstrap và cell Drive.
Tất cả lệnh train/inference còn yêu cầu `--allow-gpu`; thiếu cờ sẽ dừng trước khi nạp model.
E0 chỉ suy luận, cũng cần GPU do model 4B. Không chạy model này trên CPU.

```bash
python -m solution.run predict --allow-gpu --config configs/e0.yaml \
  --pred work/e0.jsonl --split test --batch-size 4 --resume \
  --score-out report/evidence/e0.json --name E0
```

`--resume` tiếp tục nếu bị ngắt, kiểm tra cùng config/split/model checkpoint và không trùng image_id.
Nếu file đã đầy đủ, bỏ qua nạp model và chấm lại bằng trọng tài.
Output thô được giữ nguyên trong `work/e0.jsonl`, không thay bằng nhãn GT hay sửa JSON cho tăng điểm.
Lần E0 đầu giải quyết revision model thành commit cụ thể trong manifest; các lần tiếp theo dùng đúng commit đó.

## 5. Train và kiểm tra E1, E2, E3

Notebook bước 6: đặt EXPERIMENT lần lượt `e1`, `e2`, `e3`; chạy cell train rồi bước 7
cho từng setup. Mặc định batch_size=2, grad_accum=4, 300 optimizer steps, seed=42.

```bash
python -m solution.run train --allow-gpu --config configs/e1.yaml --out work/e1 \
  --backup-dir /content/drive/MyDrive/day10_backups
```

E2/E3 thay `e1` ở cả config và out. Trên Kaggle, bỏ `--backup-dir` hoặc dùng đường dẫn lưu khác.
Thời gian 20–30 phút trong đề là ước lượng; code ghi thời gian thật và peak allocated VRAM.
`run.json` chứa phiên bản thư viện và commit model; so sánh các run trên cùng revision.
Muốn ghim ngay từ đầu, lấy revision trong `work/e0.manifest.json` và thêm `--revision <commit>`
khi train E1–E3. Không truyền token HF trực tiếp qua lệnh shell.

Kiểm tra:

- `work/e1/encoding_check.json`: 220 token ảnh với ảnh 640×360; supervised_text là JSON + im_end.
- `work/e1/train_log.jsonl`: loss/grad_norm hữu hạn, loss nhìn chung giảm (không nhất thiết giảm mọi bước).
- `work/e1/run.json`: `finished=true`, `completed_steps=300`, thông số tài nguyên là số đo thật.
- `work/e2/checkpoint-0300/projector.safetensors` và E3 phải tồn tại.

Code dừng khi loss/gradient không hữu hạn, không đếm một optimizer step bị GradScaler bỏ qua là bước thành công.
Nếu hết VRAM: chọn batch_size=1, grad_accum=8 cho **mọi** E1/E2/E3 để giữ effective batch=8.
Nếu một run đã chạy thành công với cấu hình khác, ghi rõ thay đổi hoặc chạy lại so sánh đồng nhất;
không tự động thay batch âm thầm giữa thí nghiệm.

Debug nhanh chỉ trên val, tên file riêng:

```bash
python -m solution.run predict --allow-gpu --checkpoint work/e1 \
  --split val --limit 40 --pred work/e1_debug40.jsonl --batch-size 1
```

Không dùng prediction 40 ảnh để chấm một split đầy đủ. Không lấy checkpoint train vài bước làm evidence cuối.

## 6. Suy luận val đầy đủ và test

Sau mỗi run E1–E3:

```bash
python -m solution.run predict --allow-gpu --checkpoint work/e1 \
  --split val --pred work/e1_val.jsonl --resume --batch-size 4 \
  --score-out work/e1_val_score.json --name E1
python -m solution.run predict --allow-gpu --checkpoint work/e1 \
  --split test --pred work/e1.jsonl --resume --batch-size 4 \
  --score-out report/evidence/e1.json --name E1
```

Lặp E2/E3. Code nạp base 4-bit, adapter và projector riêng, đọc coord từ checkpoint.
VRAM/thời gian train tự chuyển vào `--meta` của `scripts/score.py`.
Nếu inference OOM, giảm batch 4 xuống 1; không đổi prompt/model/toạ độ.
Nếu output bị cắt, xem `truncated_predictions` trong manifest; tăng max_new_tokens trên val
rồi dùng cùng giá trị khi test. Khi đổi cấu hình generation, dùng file dự đoán mới để không trộn output cũ/mới.
Giữ max_new_tokens nhất quán trong E4 để chỉ thay một biến.

## 7. E4: 10% so với 100% train

```bash
python -m solution.run select-e4
```

Đọc `work/configs/selection.json`: chọn E1/E2/E3 bằng val F1, không test.
Hai config `e4_data_010.yaml`, `e4_data_100.yaml` kế thừa cấu hình **thật** của checkpoint nền,
chỉ thay train_fraction (metadata tên thí nghiệm cũng đổi). Tập con shuffle seed 42, lồng trong tập đầy đủ.

```bash
python -m solution.run train --allow-gpu --config work/configs/e4_data_010.yaml \
  --out work/e4_data_010 --backup-dir /content/drive/MyDrive/day10_backups
python -m solution.run predict --allow-gpu --checkpoint work/e4_data_010 \
  --split test --pred work/e4_data_010.jsonl --resume \
  --score-out report/evidence/e4_data_010.json --name E4_data_010
```

Nhánh 100% dùng lại setup nền đã train đủ 300 bước. Cell E4 notebook copy raw prediction
vào `work/e4_data_100.jsonl` và chấm lại bằng trọng tài, giữ nguyên tài nguyên của run gốc.
Để làm bằng CLI, dùng giá trị base/checkpoint đọc từ selection:

```bash
# Ví dụ NẾU baseline chọn e2:
cp work/e2.jsonl work/e4_data_100.jsonl
python -m solution.run score --pred work/e4_data_100.jsonl --checkpoint work/e2 \
  --name E4_data_100 --out report/evidence/e4_data_100.json
```

Không cần train thêm nhánh 100%. Đây là cùng baseline, không phải một lần chạy độc lập.
300 bước giữ cố định nên tập 10% được lặp nhiều hơn; ghi giới hạn này trong phân tích.

## 8. Phân tích và báo cáo (CPU)

```bash
python -m solution.analysis
python -m solution.report
# Đọc report/REPORT_PREVIEW.md, sau đó:
python -m solution.report --write
python scripts/check_submission.py
python -m solution.package results --out work/day10_results.zip
```

Notebook làm tương tự ở bước 10, tải ZIP về Colab hoặc copy vào Drive.
Phân tích kiểm tra raw predictions khớp evidence, vẽ ít nhất sáu ảnh E1, đếm nhóm lỗi,
so cùng output E1 khi hiểu đúng/sai thang và tính CI95 hiệu F1 bằng paired bootstrap.
`REPORT.md` được điền số thật; đọc lại nhận xét, xem ảnh và bổ sung nguyên nhân lỗi bạn quan sát.
Không có cơ chế bảo đảm model đạt ngưỡng F1/parse của rubric trước khi chạy.

ZIP kết quả chứa báo cáo, evidence, ảnh, raw prediction và metadata/log/split;
không chứa ảnh dữ liệu, secrets, trọng số hay optimizer.
Giải nén ZIP kết quả vào repo local để nhận các file tương ứng. `work/`, `data/` vẫn không commit.
Không thay ZIP results cho checkpoint train: ZIP đã loại bỏ trọng số/trạng thái optimizer.

**Lưu ý test của đề:** `test_report_template_has_student_markers_and_ten_questions`
yêu cầu REPORT còn `<điền`, trong khi checker nộp bài yêu cầu bỏ `<điền`.
Vì giữ nguyên tests, chạy toàn bộ test **trước** khi render báo cáo; sau khi render,
chạy `python -m pytest -q solution/tests` và `check_submission.py`.
Nếu chạy lại tests gốc với report đã hoàn tất, đúng test template này sẽ không còn phù hợp;
không sửa tests để che điều đó.

## 9. Khôi phục sau khi ngắt session

Nếu checkpoint ở Drive, copy thư mục run đầy đủ về `work/eN`, gồm checkpoint-NNNN,
latest.json và log. Chuẩn bị lại dataset cùng dataset_version. Resume bằng config ban đầu:

```bash
python -m solution.run train --allow-gpu --config configs/e2.yaml --out work/e2 \
  --resume work/e2/checkpoint-0100 --backup-dir /content/drive/MyDrive/day10_backups
```

Nếu run trước dùng batch=1/accum=8, truyền lại `--batch-size 1 --grad-accum 8`.
Resume kiểm tra config/version, nạp adapter/projector, optimizer, schedule, scaler, RNG và vị trí sampler.
Đọc `failure.json` khi lỗi. File `training_state.pt` chỉ nạp checkpoint tin cậy do chính code tạo.
Nếu chỉ muốn suy luận, `--checkpoint` nhận cả thư mục run lẫn thư mục checkpoint cụ thể.
Evidence cuối cần checkpoint hoàn tất; checkpoint trung gian chỉ dùng debug.
E5 có checkpoint backup nhưng CLI bonus chưa hỗ trợ resume optimizer GRPO; nếu ngắt,
có thể khởi tạo lại từ SFT trong một out directory mới và ghi rõ trong báo cáo.

## 10. E5 bonus, chạy riêng nếu muốn

```bash
python -m pytest -q solution/tests/test_reward.py
# Ví dụ baseline được chọn e2:
python -m solution.run grpo --allow-gpu --checkpoint work/e2 --out work/e5 \
  --backup-dir /content/drive/MyDrive/day10_backups
python -m solution.run predict --allow-gpu --checkpoint work/e5 --pred work/e5.jsonl \
  --resume --score-out report/evidence/e5.json --name E5
```

Notebook yêu cầu thêm `RUN_E5=True`. Cấu hình 4 completions, max 128 token, 60 bước,
lr=5e-6, beta=0, use_vllm=False. Có test reward trước, lưu log và projector khi checkpoint.
TRL 1.9.2 tự cast trainable của QLoRA sang bf16 trong constructor; code chuyển lại float32
trước khi optimizer được tạo để dùng master weights cùng fp16 trên T4.
Chưa kiểm chứng E5 bằng lần chạy GPU thực tế. Có thể cần GPU nhiều VRAM hơn cho nhóm generation.
Sau khi chạy, phân tích reward hacking và so F1/CI với SFT; không tự tuyên bố có bonus khi chỉ có code.

## 11. Kiểm tra và nộp

Chỉ commit code/notebook, hướng dẫn, REPORT đã duyệt, `report/evidence/*.json`, ảnh và phân tích.
Không commit data/work/secret/trọng số. Repo hiện mới có một commit gốc; cần ≥5 commit có ý nghĩa.
Có thể chia các thay đổi hiện tại thành commit data/encoding, train/checkpoint, notebook/hướng dẫn,
sau đó commit evidence/report sau khi chạy. Chưa tự tạo commit hoặc push trong phiên này.

```bash
git status --short
python scripts/check_submission.py
```

Checker cần 0 FAIL trước khi nộp. Chạy trong repo Git local có lịch sử commit;
ZIP source không chứa `.git`, nên checker trên cloud có thể WARN Git.

## Tài liệu API đã đối chiếu

- [Qwen3-VL / Transformers](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [Config Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/config.json)
- [PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora)
- [Mã GRPO của TRL v1.9.2](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py)

Phiên bản gốc của đề được giữ nguyên, đã xác nhận tải được wheel.
Với Transformers 5.19.0, skip modules dùng đường dẫn `model.visual` và `lm_head`:
matcher khớp từ đầu tên module, nên chỉ `visual` không loại trừ `model.visual.blocks.*`.
Kiểm tra CPU không thay thế kiểm tra CUDA/NF4 và convergence trên runtime của bạn.

## Chạy một lần trên Kaggle (`lab_day10_kaggle.ipynb`)

Notebook này nhúng sẵn mã nguồn (ZIP+base64), không cần upload ZIP hay clone repo.

1. Kaggle → *New Notebook* → *File → Import Notebook* → chọn `lab_day10_kaggle.ipynb`.
2. *Settings*: Accelerator **GPU T4 ×2**, Internet **On**.
3. *Add Input → Datasets*: `road-damage-dataset-potholes-cracks-and-manholes` (lorenzoarcioni).
4. (Tuỳ chọn) *Add-ons → Secrets*: `HF_TOKEN`.
5. *Save Version → Save & Run All (Commit)*.

Hai GPU chạy song song: lane 0 = E1 → E3, lane 1 = E0 → E2; sau đó E4. Log từng lane ở `work/logs/`.
Một thí nghiệm lỗi không làm hỏng thí nghiệm khác; cell cuối in bảng tổng kết và tạo `day10_results.zip`.
Sau khi sửa bất kỳ file trong `solution/`, `scripts/`, `configs/`, sinh lại notebook:

```bash
python -m solution.make_kaggle_notebook
```

**Chạy lại sau lỗi:** nếu một thí nghiệm lỗi (ví dụ E3), thêm output của lần chạy trước làm input
(*Add Input → Notebooks → Your Work*) rồi Save & Run All. Cell "Dùng lại kết quả" khôi phục các
thí nghiệm đã hoàn chỉnh (train xong + val + test + evidence) và chỉ chạy lại phần còn thiếu.
Lỗi gặp thực tế: image Kaggle có `torchao 0.10.0` làm peft 0.21.2 ném `ImportError` khi gắn LoRA lên ViT (E3);
notebook đã gỡ `torchao` và kiểm tra LoRA trên `nn.Linear` ngay sau bước cài.

**Gradient không hữu hạn khi train E3 (fp16):** ở những bước đầu, loss scale 65536 có thể làm gradient tràn (inf).
`solution/training.py` giờ để `GradScaler` bỏ qua bước đó và giảm scale, không tính là một bước tối ưu
(vẫn đủ 300 bước thật); mỗi lần bỏ qua được ghi vào `train_log.jsonl` (`"skipped": true`) và `run.json` (`skipped_steps`).
Nếu quá 25 lần liên tiếp thì dừng và báo tên tham số có gradient không hữu hạn.
