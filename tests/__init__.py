"""
Test suite for the review scorecard, KG matching, BERT validation and the
review API.

Run from the project root:

    python -m unittest discover -s tests -t .

Importing this package points the application at a throw-away SQLite
database, so the tests never touch entity_classifier.db.
"""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_TMP = Path(tempfile.mkdtemp(prefix="entityclass-tests-"))
atexit.register(shutil.rmtree, TEST_TMP, ignore_errors=True)

os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_TMP / 'test.db'}"
