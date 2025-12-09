from openai import OpenAI
import json
from tqdm import tqdm
import datetime
import time
import os
import logging
from lightmem.memory.lightmem import LightMemory
from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig
from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant
from prompts import METADATA_GENERATE_PROMPT_locomo
import sqlite3
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# ============ Configuration ============
LOGS_ROOT = "./logs"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, RUN_TIMESTAMP)
os.makedirs(RUN_LOG_DIR, exist_ok=True)

# API Configuration
API_KEYS = [
    'your-api-key-1',
    'your-api-key-2',
    'your-api-key-3',
    'your-api-key-4',
    'your-api-key-5',
]
API_BASE_URL = ''
LLM_MODEL = 'gpt-4o-mini'

# Model Paths
LLMLINGUA_MODEL_PATH = '/path/to/llmlingua-model'
EMBEDDING_MODEL_PATH = '/path/to/embedding-model'

# Data Configuration
DATA_PATH = '/path/to/locomo10.json'
DATASET_TYPE = 'locomo'

# Qdrant Storage Directories
QDRANT_PRE_UPDATE_DIR = './qdrant_pre_update'
QDRANT_POST_UPDATE_DIR = './qdrant_post_update'

os.makedirs(QDRANT_PRE_UPDATE_DIR, exist_ok=True)
os.makedirs(QDRANT_POST_UPDATE_DIR, exist_ok=True)

# Parallel Processing Configuration
MAX_WORKERS = 5
USE_PROCESS_POOL = True


# ============ Utility Functions ============

def get_process_logger(sample_id):
    logger = logging.getLogger(f"lightmem.parallel.{sample_id}")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(
            os.path.join(RUN_LOG_DIR, f"{sample_id}.log"),
            mode='w'
        )
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


def parse_locomo_timestamp(timestamp_str):
    timestamp_str = timestamp_str.strip("()")
    try:
        dt = datetime.datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp_str


def extract_locomo_sessions(conversation_dict):
    speaker_a = conversation_dict.get('speaker_a', 'Speaker_A')
    speaker_b = conversation_dict.get('speaker_b', 'Speaker_B')
    
    session_nums = set()
    for key in conversation_dict.keys():
        if key.startswith('session_') and not key.endswith('_date_time'):
            try:
                num = int(key.split('_')[1])
                session_nums.add(num)
            except:
                continue
    
    sessions = []
    timestamps = []
    
    for num in sorted(session_nums):
        session_key = f'session_{num}'
        timestamp_key = f'{session_key}_date_time'
        
        if session_key not in conversation_dict:
            continue
            
        session_data = conversation_dict[session_key]
        timestamp = conversation_dict.get(timestamp_key, '')
        
        messages = []
        for turn in session_data:
            speaker_name = turn['speaker']
            speaker_id = 'speaker_a' if speaker_name == speaker_a else 'speaker_b'
            content = turn['text']
            if 'blip_caption' in turn and turn['blip_caption']:
                content = f"{content} (image description: {turn['blip_caption']})"
            
            messages.append({
                "role": "user",
                "content": content,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
            })
            messages.append({
                "role": "assistant",
                "content": "",
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
            })
        
        sessions.append(messages)
        timestamps.append(parse_locomo_timestamp(timestamp))
    
    return sessions, timestamps, speaker_a, speaker_b


def load_lightmem(collection_name, api_key):
    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": LLMLINGUA_MODEL_PATH,
                    "device_map": "cuda",
                    "use_llmlingua2": True,
                },
                "compress_config": {
                    "instruction": "",
                    "rate": 0.6,
                    "target_token": -1
                },
            }
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {
            "model_name": "llmlingua-2",
        },
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": LLM_MODEL,
                "api_key": api_key,
                "max_tokens": 16000,
                "openai_base_url": API_BASE_URL
            },
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": EMBEDDING_MODEL_PATH,
                "embedding_dims": 384,
                "model_kwargs": {"device": "cuda"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": f'{QDRANT_POST_UPDATE_DIR}/{collection_name}',
            }
        },
        "update": "offline",
        "logging": {
            "level": "DEBUG",
            "file_enabled": True,
            "log_dir": RUN_LOG_DIR,
        }
    }
    
    lightmem = LightMemory.from_config(config)
    return lightmem


def collection_entry_count(collection_name, base_dir):
    try:
        cfg = QdrantConfig(
            collection_name=collection_name,
            path=base_dir,
            embedding_model_dims=384,
            on_disk=True,
        )
        q = Qdrant(cfg)
        try:
            points = q.get_all(with_vectors=False, with_payload=False)
            if points:
                return len(points)
        except Exception:
            pass

        storage_sqlite = os.path.join(
            base_dir, collection_name, 'collection', collection_name, 'storage.sqlite'
        )
        if not os.path.exists(storage_sqlite):
            return 0

        try:
            conn = sqlite3.connect(storage_sqlite)
            cur = conn.execute("SELECT count(*) FROM points")
            row = cur.fetchone()
            conn.close()
            if row:
                return int(row[0])
            return 0
        except Exception:
            return -1
    except Exception:
        storage_sqlite = os.path.join(
            base_dir, collection_name, 'collection', collection_name, 'storage.sqlite'
        )
        if os.path.exists(storage_sqlite):
            try:
                conn = sqlite3.connect(storage_sqlite)
                cur = conn.execute("SELECT count(*) FROM points")
                row = cur.fetchone()
                conn.close()
                if row:
                    return int(row[0])
                return 0
            except Exception:
                return -1
        return -1


# ============ Core Processing Function ============

def process_single_sample(sample, api_key):
    sample_id = sample['sample_id']
    logger = get_process_logger(sample_id)
    
    try:
        logger.info(f"{'='*70}")
        logger.info(f"[Worker {mp.current_process().name}] Processing: {sample_id}")
        logger.info(f"[Worker {mp.current_process().name}] Using API Key: {api_key[:20]}...")
        logger.info(f"{'='*70}")
        
        conversation = sample['conversation']
        sessions, timestamps, speaker_a, speaker_b = extract_locomo_sessions(conversation)
        
        logger.info(f"  Sessions: {len(sessions)}")
        logger.info(f"  Speakers: {speaker_a}, {speaker_b}")
        logger.info(f"\n{'─'*70}")
        logger.info("Phase 1: Building memory (add_memory)")
        logger.info(f"{'─'*70}")
        
        lightmem = load_lightmem(collection_name=sample_id, api_key=api_key)
        
        initial_stats = lightmem.get_token_statistics()
        case_start_time = time.time()
        add_memory_start_time = time.time()
        
        initial_add_tokens = initial_stats['llm']['add_memory']['total_tokens']
        initial_add_calls = initial_stats['llm']['add_memory']['calls']
        
        # Process each session turn by turn
        for session, timestamp in zip(sessions, timestamps):
            while session and session[0]["role"] != "user":
                session.pop(0)
            num_turns = len(session) // 2
            for turn_idx in range(num_turns):
                turn_messages = session[turn_idx*2 : turn_idx*2 + 2]
                if len(turn_messages) < 2 or turn_messages[0]["role"] != "user" or turn_messages[1]["role"] != "assistant":
                    continue
                for msg in turn_messages:
                    msg["time_stamp"] = timestamp
                is_last_turn = (session is sessions[-1] and turn_idx == num_turns - 1)
                lightmem.add_memory(
                    messages=turn_messages,
                    METADATA_GENERATE_PROMPT=METADATA_GENERATE_PROMPT_locomo,
                    force_segment=is_last_turn,
                    force_extract=is_last_turn,
                )
        
        add_memory_end_time = time.time()
        add_memory_duration = add_memory_end_time - add_memory_start_time
        
        add_memory_stats = lightmem.get_token_statistics()
        case_add_tokens = add_memory_stats['llm']['add_memory']['total_tokens'] - initial_add_tokens
        case_add_calls = add_memory_stats['llm']['add_memory']['calls'] - initial_add_calls
        case_add_prompt = add_memory_stats['llm']['add_memory']['prompt_tokens'] - initial_stats['llm']['add_memory']['prompt_tokens']
        case_add_completion = add_memory_stats['llm']['add_memory']['completion_tokens'] - initial_stats['llm']['add_memory']['completion_tokens']
        
        after_add_count = collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR)
        logger.info(f"✓ Add_memory completed: {after_add_count} entries in {add_memory_duration:.2f}s")
        logger.info(f"\n{'─'*70}")
        logger.info("Phase 2: Backing up pre-update state")
        logger.info(f"{'─'*70}")
        
        source_dir = f'{QDRANT_POST_UPDATE_DIR}/{sample_id}'
        backup_dir = f'{QDRANT_PRE_UPDATE_DIR}/{sample_id}'
        
        backup_start_time = time.time()
        
        if os.path.exists(backup_dir):
            logger.info(f"  Removing existing backup...")
            shutil.rmtree(backup_dir)
        
        logger.info(f"  Copying: {source_dir} -> {backup_dir}")
        shutil.copytree(source_dir, backup_dir)
        
        backup_end_time = time.time()
        backup_duration = backup_end_time - backup_start_time
        
        pre_update_count = collection_entry_count(sample_id, QDRANT_PRE_UPDATE_DIR)
        logger.info(f"✓ Backup completed: {pre_update_count} entries in {backup_duration:.2f}s")
        
        logger.info(f"\n{'─'*70}")
        logger.info("Phase 3: Performing offline update")
        logger.info(f"{'─'*70}")
        
        update_start_stats = lightmem.get_token_statistics()
        initial_update_tokens = update_start_stats['llm']['update']['total_tokens']
        initial_update_calls = update_start_stats['llm']['update']['calls']
        
        update_start_time = time.time()
        lightmem.construct_update_queue_all_entries()
        lightmem.offline_update_all_entries(score_threshold=0.9)
        update_end_time = time.time()
        update_duration = update_end_time - update_start_time
        
        update_end_stats = lightmem.get_token_statistics()
        case_update_tokens = update_end_stats['llm']['update']['total_tokens'] - initial_update_tokens
        case_update_calls = update_end_stats['llm']['update']['calls'] - initial_update_calls
        case_update_prompt = update_end_stats['llm']['update']['prompt_tokens'] - update_start_stats['llm']['update']['prompt_tokens']
        case_update_completion = update_end_stats['llm']['update']['completion_tokens'] - update_start_stats['llm']['update']['completion_tokens']
        
        post_update_count = collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR)
        logger.info(f"✓ Update completed: {post_update_count} entries in {update_duration:.2f}s")
        
        case_end_time = time.time()
        case_total_duration = case_end_time - case_start_time
        
        # Log summary statistics
        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY: {sample_id}")
        logger.info(f"{'='*70}")
        
        logger.info(f"\n[Storage Information]")
        logger.info(f"  Pre-update:  {QDRANT_PRE_UPDATE_DIR}/{sample_id} ({pre_update_count} entries)")
        logger.info(f"  Post-update: {QDRANT_POST_UPDATE_DIR}/{sample_id} ({post_update_count} entries)")
        logger.info(f"  Change:      {post_update_count - pre_update_count:+d} entries")
        
        logger.info(f"\n[Time Statistics]")
        logger.info(f"  Total:       {case_total_duration:.2f}s")
        logger.info(f"  ├─ Add:      {add_memory_duration:.2f}s ({add_memory_duration/case_total_duration*100:.1f}%)")
        logger.info(f"  ├─ Backup:   {backup_duration:.2f}s ({backup_duration/case_total_duration*100:.1f}%)")
        logger.info(f"  └─ Update:   {update_duration:.2f}s ({update_duration/case_total_duration*100:.1f}%)")
        
        logger.info(f"\n[Token Statistics - Add Memory]")
        logger.info(f"  Calls:       {case_add_calls}")
        logger.info(f"  Prompt:      {case_add_prompt:,}")
        logger.info(f"  Completion:  {case_add_completion:,}")
        logger.info(f"  Total:       {case_add_tokens:,}")
        
        logger.info(f"\n[Token Statistics - Update]")
        logger.info(f"  Calls:       {case_update_calls}")
        logger.info(f"  Prompt:      {case_update_prompt:,}")
        logger.info(f"  Completion:  {case_update_completion:,}")
        logger.info(f"  Total:       {case_update_tokens:,}")
        
        logger.info(f"\n[Total Usage]")
        logger.info(f"  API Calls:   {case_add_calls + case_update_calls}")
        logger.info(f"  Tokens:      {case_add_tokens + case_update_tokens:,}")
        logger.info(f"{'='*70}\n")
        
        return {
            'sample_id': sample_id,
            'status': 'success',
            'pre_update_count': pre_update_count,
            'post_update_count': post_update_count,
            'total_duration': case_total_duration,
            'add_memory_duration': add_memory_duration,
            'backup_duration': backup_duration,
            'update_duration': update_duration,
            'add_tokens': case_add_tokens,
            'add_calls': case_add_calls,
            'update_tokens': case_update_tokens,
            'update_calls': case_update_calls,
        }
        
    except Exception as e:
        logger.error(f"✗ {sample_id} failed: {str(e)}", exc_info=True)
        return {
            'sample_id': sample_id,
            'status': 'failed',
            'error': str(e)
        }


# ============ Main Execution ============

def main():
    main_logger = logging.getLogger("lightmem.parallel.main")
    main_logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(os.path.join(RUN_LOG_DIR, "main.log"), mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    main_logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    main_logger.addHandler(ch)
    
    # Log configuration
    main_logger.info("=" * 70)
    main_logger.info("PARALLEL MEMORY BUILDING")
    main_logger.info("=" * 70)
    main_logger.info(f"Workers:         {MAX_WORKERS}")
    main_logger.info(f"API Keys:        {len(API_KEYS)}")
    main_logger.info(f"Executor:        {'ProcessPool' if USE_PROCESS_POOL else 'ThreadPool'}")
    main_logger.info(f"Post-update dir: {QDRANT_POST_UPDATE_DIR}")
    main_logger.info(f"Pre-update dir:  {QDRANT_PRE_UPDATE_DIR}")
    main_logger.info("=" * 70)
    
    data = json.load(open(DATA_PATH, "r"))
    main_logger.info(f"\nLoaded {len(data)} samples from dataset")
    
    main_logger.info("\n" + "=" * 70)
    main_logger.info("Scanning existing collections...")
    main_logger.info("=" * 70)
    
    missing = []
    for sample in data:
        sample_id = sample['sample_id']
        
        pre_update_dir = f'{QDRANT_PRE_UPDATE_DIR}/{sample_id}'
        post_update_dir = f'{QDRANT_POST_UPDATE_DIR}/{sample_id}'
        
        pre_exists = os.path.exists(pre_update_dir)
        post_exists = os.path.exists(post_update_dir)
        
        if pre_exists and post_exists:
            pre_count = collection_entry_count(sample_id, QDRANT_PRE_UPDATE_DIR)
            post_count = collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR)
            
            if pre_count > 0 and post_count > 0:
                main_logger.info(
                    f"✓ {sample_id}: Complete "
                    f"(pre={pre_count}, post={post_count})"
                )
                continue
        
        status = []
        if not pre_exists:
            status.append("pre_missing")
        elif collection_entry_count(sample_id, QDRANT_PRE_UPDATE_DIR) <= 0:
            status.append("pre_empty")
            
        if not post_exists:
            status.append("post_missing")
        elif collection_entry_count(sample_id, QDRANT_POST_UPDATE_DIR) <= 0:
            status.append("post_empty")
        
        main_logger.info(f"✗ {sample_id}: Needs processing ({', '.join(status)})")
        missing.append(sample)
    
    main_logger.info(f"\nScan complete: {len(missing)}/{len(data)} samples need processing\n")
    
    if not missing:
        main_logger.info("All samples complete. Exiting.")
        return
    
    main_logger.info("=" * 70)
    main_logger.info(f"Processing {len(missing)} samples in parallel")
    main_logger.info("=" * 70)
    main_logger.info("\nAPI Key assignment:")
    for idx, sample in enumerate(missing):
        api_key_idx = idx % len(API_KEYS)
        api_key = API_KEYS[api_key_idx]
        main_logger.info(
            f"  Sample [{idx}] {sample['sample_id'][:30]}... "
            f"→ API Key [{api_key_idx}] ({api_key[:20]}...)"
        )
    main_logger.info("")
    
    start_time = time.time()
    results = []
    failed_samples = []
    
    ExecutorClass = ProcessPoolExecutor if USE_PROCESS_POOL else ThreadPoolExecutor
    
    with ExecutorClass(max_workers=MAX_WORKERS) as executor:
        future_to_sample = {}
        for idx, sample in enumerate(missing):
            api_key_idx = idx % len(API_KEYS)
            api_key = API_KEYS[api_key_idx]
            
            future = executor.submit(process_single_sample, sample, api_key)
            future_to_sample[future] = sample
        
        # Process results as they complete
        with tqdm(total=len(missing), desc="Building memories") as pbar:
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'success':
                        pbar.set_postfix_str(f"✓ {result['sample_id']}")
                    else:
                        failed_samples.append(result['sample_id'])
                        pbar.set_postfix_str(f"✗ {result['sample_id']}")
                        
                except Exception as e:
                    main_logger.error(f"Unexpected error for {sample['sample_id']}: {e}", exc_info=True)
                    failed_samples.append(sample['sample_id'])
                
                pbar.update(1)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    main_logger.info("\n" + "=" * 70)
    main_logger.info("PROCESSING COMPLETE")
    main_logger.info("=" * 70)
    
    successful = [r for r in results if r['status'] == 'success']
    
    main_logger.info(f"\n[Overall Statistics]")
    main_logger.info(f"  Total samples:   {len(missing)}")
    main_logger.info(f"  Successful:      {len(successful)}")
    main_logger.info(f"  Failed:          {len(failed_samples)}")
    main_logger.info(f"  Wall time:       {total_duration:.2f}s ({total_duration/60:.2f} min)")
    
    if successful:
        avg_duration = sum(r['total_duration'] for r in successful) / len(successful)
        total_tokens = sum(r['add_tokens'] + r['update_tokens'] for r in successful)
        total_calls = sum(r['add_calls'] + r['update_calls'] for r in successful)
        
        main_logger.info(f"\n[Performance Metrics]")
        main_logger.info(f"  Avg per sample:  {avg_duration:.2f}s")
        main_logger.info(f"  Speedup:         {avg_duration * len(successful) / total_duration:.2f}x")
        main_logger.info(f"  Total API calls: {total_calls}")
        main_logger.info(f"  Total tokens:    {total_tokens:,}")
    
    if failed_samples:
        main_logger.info(f"\n[Failed Samples]")
        for sample_id in failed_samples:
            main_logger.info(f"  - {sample_id}")
    
    main_logger.info(f"\n{'='*70}")
    main_logger.info(f"Pre-update:  {QDRANT_PRE_UPDATE_DIR}")
    main_logger.info(f"Post-update: {QDRANT_POST_UPDATE_DIR}")
    main_logger.info(f"Logs:        {RUN_LOG_DIR}")
    main_logger.info("=" * 70)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()