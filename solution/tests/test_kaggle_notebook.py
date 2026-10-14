import ast
import base64
import io
import json
import re
import zipfile
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'lab_day10_kaggle.ipynb'
pytestmark = pytest.mark.skipif(not NOTEBOOK.exists(), reason='Kaggle notebook is not part of this ZIP')


def _payload(notebook):
    match = re.search(r'"""\n(.*?)\n"""', notebook.cells[1].source, re.S)
    return zipfile.ZipFile(io.BytesIO(base64.b64decode(match.group(1))))


def test_kaggle_notebook_valid_unexecuted_and_without_secrets():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert cell.execution_count is None and not cell.outputs
    text = NOTEBOOK.read_text(encoding='utf-8')
    assert 'google.colab' not in text and 'drive.mount' not in text
    assert 'RUN_E5 = False' in text
    assert not re.search(r"KAGGLE_KEY'?\]?\s*=\s*['\"][0-9a-f]{20,}", text)


def test_embedded_source_is_not_stale():
    archive = _payload(nbformat.read(NOTEBOOK, as_version=4))
    names = set(archive.namelist())
    assert {'solution/run.py', 'scripts/prepare_data.py', 'configs/e1.yaml', 'requirements-colab.txt'} <= names
    stale = [n for n in names if n.endswith(('.py', '.yaml', '.txt')) and (ROOT / n).read_bytes() != archive.read(n)]
    assert not stale, f'Run `python -m solution.make_kaggle_notebook` again: {stale}'
