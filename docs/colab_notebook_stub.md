# SleepTrain v2 Colab Stub

```python
!pip install psycopg2-binary sentence-transformers torch certifi

# Optional: wandb login
import wandb
# wandb.login()

import os
os.environ["COCOINDEX_DB_URL"] = "postgresql://<user>:<password>@<host>:<port>/sleeptrain"
os.environ["REQUESTS_CA_BUNDLE"] = __import__("certifi").where()

from scripts.memory.coco_memory_flow import COCOIndexMemoryFlow

flow = COCOIndexMemoryFlow(db_url=os.environ["COCOINDEX_DB_URL"])
flow.upsert({"person_id": "demo", "fact": "Demo fact", "importance": 5, "type": "fact"})
print(flow.query("Demo fact", top_k=1))
```

Replace DB URL with your connection, and ensure the database is reachable from Colab (use a secure tunnel or public endpoint). You can also load checkpoints and run benchmarks using `scripts/evaluation/benchmarks.py` and generate reports via `scripts/analysis/generate_benchmark_report.py`.
