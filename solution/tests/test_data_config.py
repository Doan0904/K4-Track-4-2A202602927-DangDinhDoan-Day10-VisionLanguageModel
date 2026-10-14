import json
import sys

import pytest

from solution.config import create_e4, load_config
from solution.data import BatchStream, objects_to_answer, prompt, subset_train, yolo_to_box
from solution.modeling import LLM_TARGET, VIT_TARGET
from solution.run import main


def test_coordinate_roundtrip_and_empty_answer():
    obj = yolo_to_box('pothole', [.5, .5, .2, .4])
    assert obj['bbox_2d'] == [400, 300, 600, 700]
    pixel = yolo_to_box('pothole', [.5, .5, .2, .4], 'pixel')
    assert pixel['bbox_2d'] == [256, 108, 384, 252]
    assert objects_to_answer([]) == '[]'
    assert '0–1000' in prompt() and '640x360' in prompt('pixel')


def test_sorting_clipping_and_invalid_box():
    boxes = json.loads(objects_to_answer([
        {'label': 'crack', 'yolo': [.8, .8, .1, .1]},
        {'label': 'pothole', 'yolo': [.05, .05, .2, .2]}]))
    assert boxes[0]['label'] == 'pothole'
    assert boxes[0]['bbox_2d'] == [0, 0, 150, 150]
    with pytest.raises(ValueError):
        yolo_to_box('pothole', [.5, .5, 0, .1])
    with pytest.raises(ValueError):
        yolo_to_box('pothole', [float('nan'), .5, .1, .1])


def test_unusable_boxes_are_excluded_without_inventing_geometry():
    valid = {'label': 'pothole', 'yolo': [.5, .5, .2, .2]}
    invalid = [{'label': 'crack', 'yolo': [.5, .5, 0, .1]},
               {'label': 'manhole', 'yolo': [.5, .5, .1, 0]},
               {'label': 'crack', 'yolo': [.5, .5, .00001, .00001]}]
    assert objects_to_answer([valid]+invalid) == objects_to_answer([valid])
    assert objects_to_answer(invalid) == '[]'
    assert invalid[0]['yolo'][2] == 0


def test_e4_changes_only_data_fraction(tmp_path):
    paths = create_e4('configs/e2.yaml', tmp_path)
    a, b = (load_config(p) for p in paths)
    for key in ('train_fraction', 'ablation'):
        a.pop(key)
        b.pop(key)
    assert a == b
    rows = [{'image_id': str(i)} for i in range(50)]
    small, large = subset_train(rows, .1), subset_train(rows)
    assert small == large[:5]
    assert len({r['image_id'] for r in large}) == 50


def test_resume_stream_matches_uninterrupted_run():
    stream = BatchStream(5, 2, 42)
    for _ in range(7):
        stream.next_indices()
    state = stream.state_dict()
    resumed = BatchStream(5, 2, 42)
    resumed.load_state_dict(state)
    assert [stream.next_indices() for _ in range(20)] == [resumed.next_indices() for _ in range(20)]


def test_module_selection_excludes_projectors_and_wrong_names():
    llm = [f'model.language_model.layers.{i}.{part}' for i in range(36)
           for part in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                        'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')]
    vit = [f'model.visual.blocks.{i}.{part}' for i in range(24)
           for part in ('attn.qkv', 'attn.proj', 'mlp.linear_fc1', 'mlp.linear_fc2')]
    assert sum(bool(LLM_TARGET.fullmatch(n)) for n in llm+vit) == 252
    assert sum(bool(VIT_TARGET.fullmatch(n)) for n in llm+vit) == 96
    assert not VIT_TARGET.fullmatch('model.visual.merger.linear_fc1')
    assert not LLM_TARGET.fullmatch('model.visual.blocks.0.mlp.up_proj')


@pytest.mark.parametrize('command', [
    ['train', '--out', 'work/must-not-run'],
    ['predict', '--pred', 'work/must-not-run.jsonl'],
    ['grpo', '--checkpoint', 'work/nonexistent', '--out', 'work/must-not-run'],
])
def test_gpu_commands_require_explicit_flag(command):
    with pytest.raises(SystemExit) as error:
        main(command)
    assert error.value.code == 2


def test_help_and_imports_do_not_import_heavy_libraries():
    import subprocess
    result = subprocess.run([sys.executable, '-c',
        'import sys; import solution.run, solution.training, solution.inference, solution.grpo; '
        'assert not any(m in sys.modules for m in ("torch", "transformers", "peft", "trl"))'],
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
