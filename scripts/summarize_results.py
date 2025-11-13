import os
import json
import argparse
from typing import Dict, List

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_result_files(results_dir: str) -> List[str]:
    files = []
    for name in os.listdir(results_dir):
        if name.startswith('result_') and name.endswith('.json'):
            files.append(os.path.join(results_dir, name))
    return sorted(files)

def summarize(results_dir: str, qdrant_dir: str) -> Dict:
    files = list_result_files(results_dir)
    total = 0
    correct_sum = 0
    times: List[float] = []
    vector_counts: Dict[str, int] = {}
    missing: List[str] = []

    # lazy import to avoid dependency issues when only summarizing accuracy/time
    try:
        from qdrant_client import QdrantClient
    except Exception:
        QdrantClient = None

    for fp in files:
        try:
            data = load_json(fp)
        except Exception:
            continue
        qid = str(data.get('question_id') or '')
        if not qid:
            # derive from filename: result_<qid>.json
            base = os.path.basename(fp)
            qid = base[len('result_'):-len('.json')]
        total += 1
        try:
            correct_sum += int(data.get('correct') or 0)
        except Exception:
            pass
        ct = data.get('construction_time')
        try:
            if isinstance(ct, (int, float)):
                times.append(float(ct))
        except Exception:
            pass

        # count vectors via embedded qdrant
        if QdrantClient is not None:
            collection_path = os.path.join(qdrant_dir, qid)
            try:
                client = QdrantClient(path=collection_path)
                # collection name equals question_id
                res = client.count(collection_name=qid, exact=True)
                # CountResult or int depending on version
                count_val = getattr(res, 'count', res)
                vector_counts[qid] = int(count_val)
            except Exception:
                vector_counts[qid] = 0
                missing.append(qid)
        else:
            vector_counts[qid] = 0

    accuracy = (correct_sum / total) if total else 0.0
    avg_time = (sum(times) / len(times)) if times else 0.0
    total_vectors = sum(vector_counts.values())
    avg_vectors = (total_vectors / len(vector_counts)) if vector_counts else 0.0
    top = sorted(vector_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_list = [{'question_id': k, 'vector_count': v} for k, v in top]

    return {
        'total_samples': total,
        'correct_count': correct_sum,
        'accuracy': accuracy,
        'avg_construction_time': avg_time,
        'total_vectors': total_vectors,
        'avg_vectors_per_collection': avg_vectors,
        'vector_counts': vector_counts,
        'top_collections_by_vectors': top_list,
        'missing_collections': missing,
        'source': {'results_dir': results_dir, 'qdrant_dir': qdrant_dir},
    }

def main():
    parser = argparse.ArgumentParser(description='Summarize LightMem results and vector counts')
    parser.add_argument('--results-dir', default=os.path.abspath(os.path.join('..', 'results')))
    parser.add_argument('--qdrant-dir', default=os.path.abspath(os.path.join('.', 'qdrant_data')))
    parser.add_argument('--out', default=os.path.abspath(os.path.join('.', 'reports', 'summary.json')))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary = summarize(args.results_dir, args.qdrant_dir)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"total_samples={summary['total_samples']}")
    print(f"correct_count={summary['correct_count']}")
    print(f"accuracy={summary['accuracy']:.4f}")
    print(f"avg_construction_time={summary['avg_construction_time']:.3f}s")
    print(f"total_vectors={summary['total_vectors']}")
    print(f"avg_vectors_per_collection={summary['avg_vectors_per_collection']:.2f}")
    print(f"output={args.out}")

if __name__ == '__main__':
    main()

