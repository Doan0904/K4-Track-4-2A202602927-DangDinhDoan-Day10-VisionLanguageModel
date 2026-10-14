import json

import pytest

from solution.reward import grounding_reward, grpo_reward

GT = [{'label': 'pothole', 'bbox': [100, 100, 200, 200]}]
BOX = {'label': 'pothole', 'bbox_2d': [100, 100, 200, 200]}


@pytest.mark.parametrize('raw,gt,expected', [
    (json.dumps([BOX]), GT, 1.0),
    ('invalid', GT, 0.0),
    ('[]', [], 1.0),
    ('[]', GT, .2),
    (json.dumps([BOX]), [], .2),
    (json.dumps([{**BOX, 'label': 'crack'}]), GT, .2),
    (json.dumps([BOX, BOX]), GT, .6),
    (json.dumps([{**BOX, 'bbox_2d': [0, 0, 1000, 1000]}]), GT, .2),
    (json.dumps([BOX, {**BOX, 'bbox_2d': [500, 500, 600, 600]}]), GT, .6),
])
def test_reward_cases(raw, gt, expected):
    assert grounding_reward(raw, gt) == pytest.approx(expected)


def test_reward_bridge_chat_and_text():
    assert grpo_reward([json.dumps([BOX]), [{'role': 'assistant', 'content': 'invalid'}]],
                       [GT, GT]) == [1.0, 0.0]


def test_matching_is_one_to_one_for_duplicate_gt():
    assert grounding_reward(json.dumps([BOX]), GT+GT) == pytest.approx(.6)
