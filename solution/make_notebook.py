"""Build the completed notebook without executing any cell."""
import json
from pathlib import Path


def build():
    cells = []
    def md(s):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(keepends=True)})
    def code(s):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                      'source': s.splitlines(keepends=True)})
    md('# Lab Day 10 — Qwen3-VL grounding\nĐặng Đỉnh Đoàn · 2A202602927. Code hoàn chỉnh ở `solution/`. '
       'Đọc `HUONG_DAN_CHAY.md`. **RUN_GPU=False** mặc định: không tự train hoặc inference. '
       'Upload ZIP source mới từ máy bạn; notebook không dựa vào bản GitHub chưa được cập nhật.')
    md('## 0. Môi trường Colab / Kaggle\nColab: chọn T4; Kaggle: bật GPU và Internet. '
       'Chạy cell dưới để giải nén source ZIP, sau đó cài dependencies. Không dùng Run all ngay từ đầu.')
    code('''import os, sys, json, shutil, subprocess
from pathlib import Path
IN_COLAB = Path('/content').exists()
IN_KAGGLE = Path('/kaggle/working').exists()
RUN_GPU = False   # Bạn tự đổi True khi muốn chạy model/train.
RUN_E5 = False    # Bonus, bật riêng sau SFT.
CODE_ZIP = ''    # Kaggle: đường dẫn ZIP code trong /kaggle/input; Colab để rỗng để upload.
REPO_DIR = Path.cwd() if Path('solution/run.py').exists() else Path('/content/day10' if IN_COLAB else '/kaggle/working/day10')
if not (REPO_DIR / 'solution/run.py').exists():
    if not CODE_ZIP and IN_COLAB:
        from google.colab import files
        uploaded = files.upload()  # Chọn day10_colab.zip tạo trên máy bạn.
        names = [n for n in uploaded if n.endswith('.zip')]
        if len(names) != 1:
            raise ValueError('Chọn đúng một ZIP source của bài lab')
        CODE_ZIP = names[0]
    if not CODE_ZIP:
        raise ValueError('Đặt CODE_ZIP đến ZIP source trong Kaggle Input')
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(CODE_ZIP, REPO_DIR)
os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
print('Repo:', REPO_DIR, '| GPU execution:', RUN_GPU)

def run(args, gpu=False):
    args = [str(x) for x in args]
    if gpu and not RUN_GPU:
        print('CHƯA CHẠY GPU:', ' '.join(args))
        return False
    subprocess.run(args, check=True)
    return True
''')
    code('''# Cài môi trường: không nạp model, không train.
run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-colab.txt'])
run([sys.executable, '-m', 'pip', 'check'])
# Sau khi cài, restart session/kernel nếu đã import torch/transformers rồi chạy lại cell bootstrap.
''')
    code('''BACKUP_DIR = None
MOUNT_DRIVE = IN_COLAB  # Nếu không dùng Drive, đổi False.
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    BACKUP_DIR = Path('/content/drive/MyDrive/day10_backups')
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
# Kaggle: checkpoint ở /kaggle/working; Save Version hoặc tải output trước khi kết thúc session.
''')
    md('## 1. Dữ liệu: chuẩn bị split cố định và kiểm tra nhãn\nChọn `source`, `zip` hoặc `kaggle`. '
       'Nếu dùng source/zip không cần nhập Kaggle Secrets. Đây là tác vụ CPU.')
    code('''DATA_MODE = 'kaggle' if IN_COLAB else 'source'
DATA_SOURCE = '/kaggle/input/road-damage-dataset-potholes-cracks-and-manholes'
DATA_ZIP = ''
if not Path('data/splits/meta.json').exists():
    if DATA_MODE == 'kaggle':
        if IN_COLAB:
            from google.colab import userdata
            for key in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
                os.environ[key] = userdata.get(key)
        elif IN_KAGGLE:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            for key in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
                os.environ[key] = secrets.get_secret(key)
        args = ['--kaggle']
    elif DATA_MODE == 'source':
        args = ['--source', DATA_SOURCE]
    elif DATA_MODE == 'zip':
        args = ['--zip', DATA_ZIP]
    else:
        raise ValueError(DATA_MODE)
    run([sys.executable, 'scripts/prepare_data.py', *args])
run([sys.executable, '-m', 'solution.run', 'audit'])
run([sys.executable, '-m', 'solution.analysis', '--labels-only'])
run([sys.executable, '-m', 'pytest', '-q', 'tests', 'solution/tests'])
''')
    md('## 2. Nhãn YOLO → JSON 0–1000\nXem sáu ảnh GT và đối chiếu toạ độ gốc; box pixel được dựng từ cùng nhãn chuẩn hoá.')
    code('''from scripts.labcommon import read_jsonl
from solution.data import objects_to_answer, prompt
from IPython.display import display
from PIL import Image
train_rows = read_jsonl('data/splits/train.jsonl')
print(prompt())
print(objects_to_answer(train_rows[0]['objects']))
for path in sorted(Path('report/figures/labels').glob('*.png')):
    display(Image.open(path))
''')
    md('## 3. E0: zero-shot\nCell có dùng GPU. Đổi RUN_GPU=True ở cell bootstrap rồi chạy lại cell đó. '
       'Các lệnh chỉ in ra khi False. Không có training ở E0.')
    code('''run([sys.executable, '-m', 'solution.run', 'predict', '--allow-gpu',
     '--config', 'configs/e0.yaml', '--pred', 'work/e0.jsonl', '--split', 'test',
     '--batch-size', '4', '--resume', '--score-out', 'report/evidence/e0.json', '--name', 'E0'], gpu=True)
''')
    md('## 4. Chat template và mask loss\n`solution/encoding.py` kiểm tra prefix, mask toàn prompt/ảnh, '
       'pad labels=-100 và nối pixel_values/image_grid_thw. Test CPU kiểm tra tensor; '
       'mỗi lần train còn kiểm tra 220 image tokens và decode JSON+im_end trên processor thật.')
    code('''run([sys.executable, '-m', 'pytest', '-q', 'solution/tests/test_encoding.py'])
# Sau E1, đọc work/e1/encoding_check.json để xác nhận processor thật.
''')
    md('## 5. Setup LoRA E1/E2/E3\nCode trong `solution/modeling.py`: exact module paths, '
       'ViT/lm_head không lượng tử; projector fp32 ở E2/E3. '
       '252 module LoRA LLM; E3 thêm 96 module ViT. Các config gốc được giữ nguyên.')
    code('''import yaml
for n in (1, 2, 3):
    print(yaml.safe_load(Path(f'configs/e{n}.yaml').read_text()))
# Nếu cần giảm VRAM, dùng BATCH_SIZE=1, GRAD_ACCUM=8 nhất quán cho mọi setup.
BATCH_SIZE = 2
GRAD_ACCUM = 4
''')
    md('## 6. Train SFT — chạy lần lượt E1, E2, E3\nĐổi EXPERIMENT rồi chạy lại cell cho mỗi setup. '
       'Mỗi lệnh chạy trong process riêng và giải phóng VRAM khi kết thúc. '
       'Không dùng checkpoint debug ít bước làm evidence cuối. Resume phải dùng đúng cấu hình cũ.')
    code('''EXPERIMENT = 'e1'   # Lần lượt: e1, e2, e3
RESUME_CHECKPOINT = ''  # Ví dụ work/e1/checkpoint-0100; chỉ checkpoint tin cậy của bạn.
args = [sys.executable, '-m', 'solution.run', 'train', '--allow-gpu',
        '--config', f'configs/{EXPERIMENT}.yaml', '--out', f'work/{EXPERIMENT}',
        '--batch-size', str(BATCH_SIZE), '--grad-accum', str(GRAD_ACCUM)]
if BACKUP_DIR:
    args += ['--backup-dir', BACKUP_DIR]
if RESUME_CHECKPOINT:
    args += ['--resume', RESUME_CHECKPOINT]
run(args, gpu=True)
if Path(f'work/{EXPERIMENT}/run.json').exists():
    print(Path(f'work/{EXPERIMENT}/run.json').read_text())
    print(Path(f'work/{EXPERIMENT}/encoding_check.json').read_text())
''')
    md('## 7. Suy luận validation rồi test E1–E3\nSau mỗi lần train, giữ EXPERIMENT tương ứng. '
       'Đánh giá val đầy đủ để chọn E4; debug 40 ảnh có tên file riêng. Test chấm một lần mỗi setup.')
    code('''for split in ('val', 'test'):
    pred = f'work/{EXPERIMENT}_val.jsonl' if split == 'val' else f'work/{EXPERIMENT}.jsonl'
    score = f'work/{EXPERIMENT}_val_score.json' if split == 'val' else f'report/evidence/{EXPERIMENT}.json'
    run([sys.executable, '-m', 'solution.run', 'predict', '--allow-gpu',
         '--checkpoint', f'work/{EXPERIMENT}', '--pred', pred, '--split', split,
         '--batch-size', '4', '--resume', '--score-out', score, '--name', EXPERIMENT.upper()], gpu=True)
''')
    md('## 8. E4 data-size ablation\nChạy sau khi đủ E1–E3 và val scores. '
       'Setup nền chọn trên val. Chỉ train nhánh 10%; nhánh 100% dùng lại baseline, '
       'cùng seed/hyperparameters, giữ 300 bước tối ưu.')
    code('''# CPU: tạo hai config có đúng một biến khác nhau.
required = [Path(f'work/e{n}_val_score.json') for n in (1, 2, 3)]
if all(p.exists() for p in required):
    run([sys.executable, '-m', 'solution.run', 'select-e4'])
    selection = json.loads(Path('work/configs/selection.json').read_text())
    print(selection)
else:
    print('Cần hoàn tất E1–E3 và validation trước E4')
''')
    code('''if Path('work/configs/selection.json').exists():
    args = [sys.executable, '-m', 'solution.run', 'train', '--allow-gpu',
            '--config', 'work/configs/e4_data_010.yaml', '--out', 'work/e4_data_010']
    if BACKUP_DIR:
        args += ['--backup-dir', BACKUP_DIR]
    run(args, gpu=True)
    run([sys.executable, '-m', 'solution.run', 'predict', '--allow-gpu',
         '--checkpoint', 'work/e4_data_010', '--pred', 'work/e4_data_010.jsonl',
         '--split', 'test', '--resume', '--score-out', 'report/evidence/e4_data_010.json',
         '--name', 'E4_data_010'], gpu=True)
    if RUN_GPU:
        selection = json.loads(Path('work/configs/selection.json').read_text())
        shutil.copy2(f"work/{selection['base']}.jsonl", 'work/e4_data_100.jsonl')
        run([sys.executable, '-m', 'solution.run', 'score', '--pred', 'work/e4_data_100.jsonl',
             '--checkpoint', selection['checkpoint'], '--name', 'E4_data_100',
             '--out', 'report/evidence/e4_data_100.json'])
''')
    md('## 9. E5 bonus (tuỳ chọn)\nReward test trên CPU. GRPO cần cả RUN_GPU=True và RUN_E5=True. '
       'Code chuẩn bị theo TRL 1.9.2; chưa chạy ở máy local. E5 checkpoint/projector được lưu riêng.')
    code('''run([sys.executable, '-m', 'pytest', '-q', 'solution/tests/test_reward.py'])
if RUN_E5 and Path('work/configs/selection.json').exists():
    selection = json.loads(Path('work/configs/selection.json').read_text())
    args = [sys.executable, '-m', 'solution.run', 'grpo', '--allow-gpu',
            '--checkpoint', selection['checkpoint'], '--out', 'work/e5']
    if BACKUP_DIR:
        args += ['--backup-dir', BACKUP_DIR]
    run(args, gpu=True)
    run([sys.executable, '-m', 'solution.run', 'predict', '--allow-gpu',
         '--checkpoint', 'work/e5', '--pred', 'work/e5.jsonl', '--resume',
         '--score-out', 'report/evidence/e5.json', '--name', 'E5'], gpu=True)
else:
    print('E5 chưa bật')
''')
    md('## 10. Phân tích, báo cáo và tải kết quả\nKhông sinh evidence giả. '
       'Chạy khi đã có đầy đủ e0–e3 và e4_data_010/e4_data_100. '
       'Đọc báo cáo tự động, thêm nhận xét từ sáu ảnh và kiểm tra rubric. '
       'check_submission còn FAIL git commits nếu ZIP chưa có lịch sử Git; kiểm tra lần cuối ở repo local sau commit.')
    code('''keys = ['e0', 'e1', 'e2', 'e3', 'e4_data_010', 'e4_data_100']
if all(Path(f'report/evidence/{k}.json').exists() for k in keys):
    run([sys.executable, '-m', 'solution.analysis'])
    run([sys.executable, '-m', 'solution.report', '--write'])
    print(Path('REPORT.md').read_text())
    # Công cụ gốc có test yêu cầu REPORT còn marker template: chạy tests trước bước render.
    result = subprocess.run([sys.executable, 'scripts/check_submission.py'], check=False)
    print('Submission checker exit:', result.returncode)
    run([sys.executable, '-m', 'solution.package', 'results', '--out', 'work/day10_results.zip'])
    if BACKUP_DIR:
        shutil.copy2('work/day10_results.zip', BACKUP_DIR / 'day10_results.zip')
    if IN_COLAB:
        from google.colab import files
        files.download('work/day10_results.zip')
else:
    print('Chưa đủ kết quả thật; không tạo bảng/evidence thay thế.')
''')
    notebook = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': []},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
        'nbformat': 4, 'nbformat_minor': 5}
    for i, cell in enumerate(cells):
        cell['id'] = f'day10-{i:02d}'
    Path('lab_day10.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')


if __name__ == '__main__':
    build()
