import argparse
import concurrent.futures
import logging
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_filter(filters: dict):
    """Construct Qdrant Filter from dict with float values."""
    conditions = []
    for key, value in filters.items():
        if isinstance(value, dict):
            gte = value.get("gte", None)
            lte = value.get("lte", None)
            if gte is None and lte is None:
                raise ValueError(f"No gte/lte found for key {key}")
            conditions.append(FieldCondition(
                key=key,
                range=Range(gte=gte, lte=lte)
            ))
        else:
            raise ValueError("Currently only numeric range filters are supported.")
    return Filter(must=conditions) if conditions else None



def construct_update_queue_all_entries(client, collection_name, top_k=20, keep_top_n=10, max_workers=8):
    all_entries, _ = client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_vectors=True,
        with_payload=True
    )
    logger.info(f"Retrieved {len(all_entries)} entries for update queue construction.")

    points_to_upsert = []

    def _update_queue(entry):
        eid = entry.id
        payload = entry.payload
        vec = entry.vector
        ts_float = payload.get("time_stamp", None)
        if vec is None or ts_float is None:
            return

        query_filter = create_filter({"time_stamp": {"lte": ts_float}})

        hits = client.query_points(
            collection_name=collection_name,
            query=vec,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )

        candidates = [{"id": h.id, "score": h.score} for h in hits.points if h.id != eid]
        candidates.sort(key=lambda x: x["score"], reverse=True)
        update_queue = candidates[:keep_top_n]

        new_payload = dict(payload)
        new_payload["update_queue"] = update_queue

        points_to_upsert.append(PointStruct(id=eid, vector=vec, payload=new_payload))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_update_queue, all_entries))

    for i in range(0, len(points_to_upsert), 100):
        batch = points_to_upsert[i:i+100]
        client.upsert(collection_name=collection_name, points=batch)


def build_candidate_sources(client, collection_name, score_threshold=0.2, write_back=False):
    """Find candidate_sources for each entry."""
    all_entries, _ = client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_vectors=False,
        with_payload=True
    )
    logger.info(f"Retrieved {len(all_entries)} entries for candidate source construction.")

    results = {}
    count = []
    no_zero = []
    nono = 0
    for entry in all_entries:
        eid = entry.id
        payload = entry.payload
        candidate_sources = []

        for other in all_entries:
            update_queue = other.payload.get("update_queue", [])
            for candidate in update_queue:
                if candidate["id"] == eid and candidate["score"] >= score_threshold:
                    candidate_sources.append(other.id)
                    break

        results[eid] = {
            "sources": candidate_sources,
            "count": len(candidate_sources)
        }
        count.append(len(candidate_sources))
        if len(candidate_sources) != 0:
            no_zero.append(len(candidate_sources)+1)
            nono += 1

    print(len(count))
    print(nono)
    print(len(no_zero))
    print(sum(no_zero)/len(no_zero))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, required=True, help="Qdrant collection name")
    parser.add_argument("--path", type=str, required=True, help="Path to local Qdrant database")
    args = parser.parse_args()

    client = QdrantClient(path=args.path)

    col_info = client.get_collection(collection_name=args.collection)
    num_points = col_info.points_count
    print(f"Collection {args.collection} has {num_points} points.")

    from datetime import datetime
    all_entries, _ = client.scroll(collection_name=args.collection, limit=10000, with_payload=True, with_vectors=True)
    for entry in tqdm(all_entries):
        ts_str = entry.payload.get("time_stamp", None)
        if not isinstance(ts_str, float):
            ts_float = datetime.fromisoformat(ts_str).timestamp()
            entry.payload["time_stamp"] = ts_float
            point = PointStruct(
                    id = entry.id,
                    vector = entry.vector,
                    payload =  entry.payload,
                )
            client.upsert(
                collection_name=args.collection,
                points = [point]
            )
    print(f"Converted {len(all_entries)} time_stamp fields to float.")

    construct_update_queue_all_entries(client, args.collection, top_k=20, keep_top_n=10)

    results = build_candidate_sources(client, args.collection, score_threshold=0.8, write_back=False)
    print(f"Candidate sources constructed for {len(results)} entries.")


if __name__ == "__main__":
    main()
