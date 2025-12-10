import os

import pytest

pytest.importorskip("psycopg2")
pytest.importorskip("sentence_transformers")

DB_URL = os.getenv("COCOINDEX_DB_URL")

if not DB_URL:
    pytest.skip("COCOINDEX_DB_URL not set; skipping DB-backed tests", allow_module_level=True)

from scripts.memory.coco_memory_flow import COCOIndexMemoryFlow  # noqa: E402


def test_upsert_and_query_returns_match():
    flow = COCOIndexMemoryFlow(db_url=DB_URL)
    rec1 = flow.upsert(
        {
            "person_id": "obama",
            "fact": "Obama was born in 1961.",
            "importance": 9,
        }
    )
    rec2 = flow.upsert(
        {
            "person_id": "musk",
            "fact": "Elon Musk founded SpaceX.",
            "importance": 8,
        }
    )

    results = flow.query("When was Obama born?", top_k=2, person_id="obama")
    flow.delete_by_ids([rec1.id, rec2.id])

    assert results, "query should return results"
    top_id = results[0][0].id
    matched_facts = {rec.id: rec.fact for rec, _ in results}
    assert matched_facts[top_id].startswith("Obama")


def test_semantic_diff_reports_best_match():
    flow = COCOIndexMemoryFlow(db_url=DB_URL)
    rec = flow.upsert(
        {
            "person_id": "tester",
            "fact": "I was born in 1961.",
            "type": "bio",
            "importance": 7,
        }
    )
    diff = flow.semantic_diff("What year were you born?", person_id="tester")
    flow.delete_by_ids([rec.id])

    assert diff["best_match_id"] is not None
    assert diff["best_match_score"] is not None
