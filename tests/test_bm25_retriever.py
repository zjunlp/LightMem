import os
import shutil
import tempfile
import unittest

from lightmem.factory.retriever.contextretriever.bm25 import BM25
from lightmem.configs.retriever.bm25 import BM25Config


class TestBM25Retriever(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        index_path = os.path.join(self.test_dir, "index.pkl")
        config = BM25Config(index_path=index_path, on_disk=True)
        self.retriever = BM25(config)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_insert_and_search(self):
        self.retriever.insert(
            docs=["Apple pie is tasty", "Banana smoothie"],
            payloads=[{"category": "fruit"}, {"category": "fruit"}],
            ids=["1", "2"],
        )

        results = self.retriever.search("apple pie", limit=2)
        self.assertTrue(results)
        self.assertEqual(set(results[0].keys()), {"id", "text", "score", "payload"})
        self.assertEqual(results[0]["id"], "1")

    def test_persistence(self):
        self.retriever.insert(docs=["Persistent apple"], payloads=[{"category": "fruit"}], ids=["a"])

        reloaded = BM25(self.retriever.config)
        results = reloaded.search("apple", limit=1)

        self.assertTrue(os.path.exists(self.retriever.config.index_path))
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "a")

    def test_delete(self):
        self.retriever.insert(docs=["Apple", "Banana"], ids=["1", "2"], payloads=[{}, {}])

        self.retriever.delete("1")
        results = self.retriever.search("Apple", limit=2)

        ids = {hit["id"] for hit in results}
        self.assertNotIn("1", ids)

    def test_payload_filtering(self):
        self.retriever.insert(
            docs=["Apple fruit snack", "Apple computer device"],
            payloads=[{"type": "food"}, {"type": "tech"}],
            ids=["food", "tech"],
        )

        filtered = self.retriever.search("apple", limit=5, filters={"type": "tech"})
        self.assertTrue(filtered)
        self.assertTrue(all(hit["payload"].get("type") == "tech" for hit in filtered))
        self.assertEqual({hit["id"] for hit in filtered}, {"tech"})


if __name__ == "__main__":
    unittest.main()
