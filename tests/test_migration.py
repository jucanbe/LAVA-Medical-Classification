"""The additive schema migration on a copy of the real database."""
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, inspect

from tests import PROJECT_ROOT, TEST_TMP

from database.connection import _add_missing_columns, _ADDED_COLUMNS
from database.models import Base, EntityReviewDB, RelationReviewDB
from routers.entity_review import _convert_to_response as entity_response
from routers.relation_review import _convert_to_response as relation_response
from sqlalchemy.orm import Session

REAL_DB = PROJECT_ROOT / "entity_classifier.db"


def pre_migration_copy() -> Path:
    """A copy of the real DB, with the new columns removed if already migrated."""
    target = Path(tempfile.mkdtemp(prefix="migration-", dir=TEST_TMP)) / "copy.db"
    source = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True)
    dest = sqlite3.connect(target)
    source.backup(dest)
    source.close()
    for table, column, _ in _ADDED_COLUMNS:
        columns = {row[1] for row in dest.execute(f"PRAGMA table_info({table})")}
        if column in columns:
            dest.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    dest.commit()
    dest.close()
    return target


def snapshot(path: Path) -> dict:
    con = sqlite3.connect(path)
    data = {}
    for table in ("entity_reviews", "relation_reviews", "pending_entities", "pending_relations", "entity_config", "relation_config", "llm_configs"):
        columns = [row[1] for row in con.execute(f"PRAGMA table_info({table})")]
        data[table] = (columns, con.execute(f"SELECT * FROM {table} ORDER BY id").fetchall())
    con.close()
    return data


@unittest.skipUnless(REAL_DB.exists(), "no existing database to migrate")
class TestMigration(unittest.TestCase):
    def test_migration_preserves_data_and_old_rows_stay_readable(self):
        path = pre_migration_copy()
        before = snapshot(path)

        engine = create_engine(f"sqlite:///{path}")
        with engine.begin() as conn:
            Base.metadata.create_all(conn)
            _add_missing_columns(conn)
        with engine.begin() as conn:  # idempotent
            _add_missing_columns(conn)

        inspector = inspect(engine)
        for table, column, _ in _ADDED_COLUMNS:
            self.assertIn(column, {c["name"] for c in inspector.get_columns(table)})

        after = snapshot(path)
        for table, (columns, rows) in before.items():
            new_columns, new_rows = after[table]
            self.assertEqual(len(new_rows), len(rows), table)
            index = [new_columns.index(c) for c in columns]
            self.assertEqual([tuple(r[i] for i in index) for r in new_rows], rows, table)

        with Session(engine) as session:
            entity_rows = session.query(EntityReviewDB).all()
            relation_rows = session.query(RelationReviewDB).all()
            self.assertEqual(len(entity_rows), len(before["entity_reviews"][1]))
            for row in entity_rows:
                body = entity_response(row)
                self.assertIsNone(body.scoring_version)
                self.assertEqual(body.score_history, [])
                if row.constraint_type_valid is not None:
                    self.assertEqual(body.constraint.type_valid, row.constraint_type_valid)
            for row in relation_rows:
                self.assertEqual(relation_response(row).overall_score, row.overall_score or 0.0)
        engine.dispose()
        shutil.rmtree(path.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
