"""Synthetic fixtures stay in pytest tmp_path; never write lab evidence into the repo."""
import ast
import json
import shutil
import zipfile
from pathlib import Path

import nbformat
import pytest
from PIL import Image

from scripts.labcommon import write_jsonl
from scripts.score import main as official_score
from solution.analysis import build_analysis, paired_ci, prediction_map
from solution.config import write_json
from solution.data import audit_data, objects_to_answer
from solution.package import package
from solution.report import render_report
from solution.run import main

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def synthetic_run(tmp_path):
    root = tmp_path
    counts, rows_by_split = {}, {}
    for split in ('train', 'val', 'test'):
        rows = []
        for i in range(6):
            iid = f'{split}_{i}'
            image_path = f'images/{iid}.jpg'
            (root/'data/images').mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (640, 360), 'gray').save(root/'data'/image_path)
            rows.append({'image_id': iid, 'image_path': image_path, 'width': 640, 'height': 360,
                         'session': split, 'objects': [{'label': 'pothole', 'yolo': [.5, .5, .2, .2]}]})
        rows_by_split[split] = rows
        counts[split] = {'images': 6, 'boxes': {'pothole': 6, 'crack': 0, 'manhole': 0}}
        write_jsonl(root/'data/splits'/f'{split}.jsonl', rows)
    version = 'synthetic-test-only'
    write_json(root/'data/splits/meta.json', {'dataset_version': version, 'counts': counts})
    for key in ('e0', 'e1', 'e2', 'e3', 'e4_data_010', 'e4_data_100'):
        write_jsonl(root/'work'/f'{key}.jsonl', [
            {'image_id': r['image_id'], 'raw': '[]' if key == 'e0' else objects_to_answer(r['objects'])}
            for r in rows_by_split['test']])
        official_score(['--pred', str(root/'work'/f'{key}.jsonl'), '--name', key,
                        '--splits-dir', str(root/'data/splits'), '--out', str(root/'report/evidence'/f'{key}.json'),
                        '--meta', 'trainable_params=1', '--meta', 'peak_vram_gb=1.0', '--meta', 'train_minutes=1.0'])
    for key in ('e1', 'e2', 'e3', 'e4_data_010'):
        cfg = {'name': key.upper(), 'coord': 'norm1000'}
        run = {'dataset_version': version, 'coord': 'norm1000', 'finished': True, 'completed_steps': 300,
               'max_steps': 300, 'trainable_params': 1, 'train_minutes': 1, 'peak_vram_gb': 1,
               'train_images': 1 if key == 'e4_data_010' else 6,
               'architecture': {'vit_backbone': 2, 'projector': 3, 'llm_including_embeddings': 4, 'total': 9}}
        write_json(root/'work'/key/'checkpoint-0300/config.json', cfg)
        write_json(root/'work'/key/'checkpoint-0300/run.json', run)
        write_json(root/'work'/key/'latest.json', {'checkpoint': 'checkpoint-0300'})
    write_json(root/'work/e1/encoding_check.json', {'image_id': 'train_0', 'image_tokens': 220,
               'prompt_tokens': 223, 'answer_tokens': 10, 'supervised_text': '[]<|im_end|>'})
    write_json(root/'work/configs/selection.json', {'base': 'e1', 'checkpoint': 'work/e1/checkpoint-0300',
               'validation_f1': 1.0, 'selection_split': 'val'})
    shutil.copy2(ROOT/'REPORT.md', root/'REPORT.md')
    return root, rows_by_split


def test_cpu_pipeline_scorer_analysis_figures_and_report(synthetic_run):
    root, _ = synthetic_run
    analysis = build_analysis(root/'data', root/'work', root/'report/evidence', root/'report/analysis.json')
    assert analysis['coordinate_control']['correct_norm1000_f1'] == 1.0
    assert analysis['coordinate_control']['wrong_pixel_f1'] == 0.0
    assert analysis['figure_count'] == 6
    assert len(list((root/'report/figures/predictions').glob('*.png'))) == 6
    assert analysis['paired']['e4_data_100-e4_data_010']['delta_ci95'] == [0.0, 0.0]
    text = render_report(root, write=True)
    assert '<điền' not in text
    assert version_in(text, 'synthetic-test-only')
    assert '100%=' in text and '10%=' in text


def version_in(text, value):
    return f'`dataset_version`: `{value}`' in text


def test_analysis_refuses_predictions_inconsistent_with_evidence(synthetic_run):
    root, rows = synthetic_run
    write_jsonl(root/'work/e1.jsonl', [{'image_id': r['image_id'], 'raw': '[]'} for r in rows['test']])
    with pytest.raises(ValueError, match='disagree with evidence'):
        build_analysis(root/'data', root/'work', root/'report/evidence', root/'report/analysis.json')


def test_session_leakage_is_rejected(synthetic_run):
    root, rows = synthetic_run
    rows['val'][0]['session'] = 'train'
    write_jsonl(root/'data/splits/val.jsonl', rows['val'])
    with pytest.raises(ValueError, match='Session leakage'):
        audit_data(root/'data')


def test_audit_reports_zero_area_without_changing_official_counts(synthetic_run):
    root, rows = synthetic_run
    rows['train'][0]['objects'][0]['yolo'][2] = 0
    write_jsonl(root/'data/splits/train.jsonl', rows['train'])
    stats = audit_data(root/'data')['splits']['train']
    assert stats['boxes']['pothole'] == 6
    assert stats['unusable_target_boxes']['pothole'] == 1
    assert stats['unusable_target_examples'][0]['reason'] == 'zero_area'
    assert len(rows['train'][0]['objects']) == 1


def test_duplicate_or_missing_predictions_are_rejected(tmp_path):
    path = tmp_path/'pred.jsonl'
    rows = [{'image_id': 'a'}, {'image_id': 'b'}]
    write_jsonl(path, [{'image_id': 'a', 'raw': '[]'}]*2)
    with pytest.raises(ValueError, match='Duplicate'):
        prediction_map(path, rows)
    write_jsonl(path, [{'image_id': 'a', 'raw': '[]'}])
    with pytest.raises(ValueError, match='complete predictions'):
        prediction_map(path, rows)


def test_paired_ci_identical_outputs_are_zero():
    rows = [{'image_id': str(i), 'width': 640, 'height': 360, 'objects': []} for i in range(8)]
    preds = {r['image_id']: '[]' for r in rows}
    result = paired_ci(rows, preds, preds, n_boot=100)
    assert result['delta_f1_b_minus_a'] == 0
    assert result['delta_ci95'] == [0.0, 0.0]
    assert not result['excludes_zero']


def test_results_archive_excludes_weights_and_secrets(synthetic_run):
    root, _ = synthetic_run
    (root/'work/.env').write_text('do-not-export')
    (root/'work/model.safetensors').write_bytes(b'not-real-weights')
    (root/'work/training_state.pt').write_bytes(b'not-real-optimizer')
    path = package('results', root/'export.zip', root)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    assert 'REPORT.md' in names and 'work/e1.jsonl' in names
    assert not any(n.endswith(('.pt', '.safetensors', '.env')) for n in names)
    assert not any(n.startswith('data/images/') for n in names)


def test_notebook_valid_unexecuted_and_gpu_opt_in():
    notebook = nbformat.read(ROOT/'lab_day10.ipynb', as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert cell.execution_count is None and not cell.outputs
    text = (ROOT/'lab_day10.ipynb').read_text()
    assert 'RUN_GPU = False' in text and 'RUN_E5 = False' in text
    assert 'YOUR CODE HERE' not in text
