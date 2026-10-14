from types import SimpleNamespace

import pytest
from PIL import Image

torch = pytest.importorskip('torch', reason='Install optional CPU torch wheel for tensor checks')
from solution.encoding import collate, encode_example, inspect_encoding


class FakeProcessor:
    """Deterministic chat-template stand-in: tests masking/collation without model weights."""
    def __init__(self, mismatch=False):
        self.mismatch = mismatch
        self.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda s: 99 if s == '<|im_end|>' else 77,
                                         decode=lambda t, **kw: 'JSON<|im_end|>')

    def apply_chat_template(self, messages, **kwargs):
        prefix = [10]+[77]*220+[11, 12]
        ids = prefix if len(messages) == 1 else prefix+[31, 32, 99, 13]
        if self.mismatch and len(messages) == 2:
            ids[0] = 999
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones((1, len(ids)), dtype=torch.long),
                'mm_token_type_ids': torch.tensor([[int(i == 77) for i in ids]]),
                'pixel_values': torch.zeros((880, 3)), 'image_grid_thw': torch.tensor([[1, 22, 40]])}


@pytest.fixture
def row(tmp_path):
    Image.new('RGB', (640, 360)).save(tmp_path/'frame.jpg')
    return tmp_path, {'image_id': 'frame', 'image_path': 'frame.jpg', 'width': 640, 'height': 360,
                     'objects': [{'label': 'pothole', 'yolo': [.5, .5, .2, .2]}]}


def test_mask_only_answer_and_im_end(row):
    root, r = row
    enc = encode_example(FakeProcessor(), r, root)
    assert (enc['labels'][0, :223] == -100).all()
    assert enc['labels'][0, 223:226].tolist() == [31, 32, 99]
    assert enc['labels'][0, 226] == -100
    assert (enc['labels'][enc['input_ids'] == 77] == -100).all()
    check = inspect_encoding(FakeProcessor(), r, root)
    assert check['image_tokens'] == 220 and check['answer_tokens'] == 3


def test_bad_prefix_and_truncation_are_rejected(row):
    root, r = row
    with pytest.raises(RuntimeError, match='prefix mismatch'):
        encode_example(FakeProcessor(mismatch=True), r, root)
    with pytest.raises(ValueError, match='do not truncate'):
        encode_example(FakeProcessor(), r, root, max_length=100)


def test_collate_masks_padding_and_concatenates_patches():
    def sample(ids):
        n = len(ids)
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones((1, n), dtype=torch.long),
                'labels': torch.tensor([ids]), 'mm_token_type_ids': torch.zeros((1, n), dtype=torch.long),
                'pixel_values': torch.zeros((4, 3)), 'image_grid_thw': torch.tensor([[1, 2, 2]])}
    output = collate([sample([1, 2]), sample([1, 2, 3, 4])], pad_token_id=0)
    assert output['input_ids'].tolist() == [[1, 2, 0, 0], [1, 2, 3, 4]]
    assert output['labels'].tolist() == [[1, 2, -100, -100], [1, 2, 3, 4]]
    assert output['attention_mask'][0].tolist() == [1, 1, 0, 0]
    assert output['pixel_values'].shape == (8, 3)
    assert output['image_grid_thw'].shape == (2, 3)
