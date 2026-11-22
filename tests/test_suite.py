import os

SKIP_ENGINE_TESTS = os.environ.get("ALPHAMIND_TEST_DATABASE", "").lower() not in ("1", "true")
DATA_ENGINE_URI = os.environ.get("DB_URI")
