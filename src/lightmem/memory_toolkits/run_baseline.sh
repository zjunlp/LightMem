# An example of running the baseline model. 
# Please modify the following variables to fit your own dataset and memory configuration. 
# ========================================================
memory_type="FullContext"
dataset_type="LoCoMo"
llm_model_type="gpt-4.1-mini"
dataset_path="/your/path/to/dataset/locomo/locomo10.json"
config_path="/your/path/to/LightMem/src/lightmem/memory_toolkits/memories/configs/FullContext.json"
num_workers=1  # We suggest to set num_workers=1.
tokenizer_path="gpt-4.1-mini"
# log_dir="FullContext_${llm_model_type}_LoCoMo_construction_logs"
# log_dir="FullContext_${llm_model_type}_LoCoMo_search_logs"
log_dir="FullContext_${llm_model_type}_LoCoMo_eval_logs"

token_cost_prefix="token_cost"
# pid_prefix="process_construction"
# pid_prefix="process_search"
pid_prefix="process_eval"
# pid_prefix="process_analyze" 
top_k=-1
ranges=(
    "0 1"
    # "1 2"
    # "2 3"
    # "3 4"
    # "4 5"
    # "5 6"
    # "6 7"
    # "7 8"
    # "8 9"
    # "9 10"
)
api_keys=(
    ""
    ""
    ""
    ""
    ""
)
base_urls=(
    ""
    ""
    ""
    ""
)
api_keys_for_image=(
    ""
    ""
    ""
    ""
    ""
)
base_urls_for_image=(
    ""
    ""
    ""
    ""
    ""
)
# ========================================================

[ ! -d "$log_dir" ] && mkdir -p "$log_dir"

for ((i=0; i<${#ranges[@]}; i++)); do
    read start_idx end_idx <<< "${ranges[$i]}"
    search_results_file="${memory_type}_${llm_model_type}_${dataset_type}_${top_k}_${start_idx}_${end_idx}.json"
    evaluation_results_file="${memory_type}_${llm_model_type}_${dataset_type}_${top_k}_${start_idx}_${end_idx}_evaluation.json"
    
    # ========== Only MemZeroGraph / MemZero need MEM0_DIR ==========
    if [[ "$memory_type" == "MemZeroGraph" || "$memory_type" == "MemZero" || "$memory_type" == "NaiveRAG" ]]; then
        mem0_worker_dir="/your/path/to/LightMem/src/lightmem/memory_toolkits/mem0_worker_dir_${memory_type,,}_${llm_model_type}/mem0_worker_$((i+1))"
        mkdir -p "$mem0_worker_dir"
        export MEM0_DIR="$mem0_worker_dir"
    else
        unset MEM0_DIR
    fi

    export HF_ENDPOINT="https://hf-mirror.com"
    export OPENAI_API_KEY="${api_keys[$i]}" 
    export OPENAI_API_BASE="${base_urls[$i]}"
    export OPENAI_API_KEY_FOR_IMAGE="${api_keys_for_image[$i]}" 
    export OPENAI_API_BASE_FOR_IMAGE="${base_urls_for_image[$i]}"

    log_file="${log_dir}/${pid_prefix}_$((i+1))_${start_idx}_${end_idx}.log"
    token_cost_file="${token_cost_prefix}_${memory_type,,}_${llm_model_type}_$((i+1))_${start_idx}_${end_idx}"
    pid_file="${log_dir}/${pid_prefix}_$((i+1)).pid"

    [ ! -f "$log_file" ] && touch "$log_file"
    # nohup python memory_construction.py \
    #     --memory-type "$memory_type" \
    #     --dataset-type "$dataset_type" \
    #     --dataset-path "$dataset_path" \
    #     --config-path "$config_path" \
    #     --num-workers "$num_workers" \
    #     --start-idx "$start_idx" \
    #     --end-idx "$end_idx" \
    #     --token-cost-save-filename "$token_cost_file" \
    #     --tokenizer-path "$tokenizer_path" \
    #     --rerun \
    #     --message-preprocessor "memories.datasets.locomo_preprocessor:NaiveRAG_style_message_for_LoCoMo" \
    #     > "$log_file" 2>&1 &

    # nohup python memory_construction.py \
    #     --memory-type "$memory_type" \
    #     --dataset-type "$dataset_type" \
    #     --dataset-path "$dataset_path" \
    #     --config-path "$config_path" \
    #     --num-workers "$num_workers" \
    #     --start-idx "$start_idx" \
    #     --end-idx "$end_idx" \
    #     --token-cost-save-filename "$token_cost_file" \
    #     --tokenizer-path "$tokenizer_path" \
    #     --rerun \
    #     > "$log_file" 2>&1 &
    
    # top-k = -1 means retrieve all memories, for full-context baseline
    # nohup python memory_search.py \
    #     --memory-type "$memory_type" \
    #     --dataset-type "$dataset_type" \
    #     --dataset-path "$dataset_path" \
    #     --config-path "$config_path" \
    #     --num-workers "$num_workers" \
    #     --start-idx "$start_idx" \
    #     --end-idx "$end_idx" \
    #     --strict \
    #     --top-k -1 \
    #     > "$log_file" 2>&1 &
    
    # nohup python memory_evaluation.py \
    #     --search-results-path "$search_results_file" \
    #     --dataset-type "$dataset_type" \
    #     --qa-model "gpt-4.1-mini" \
    #     --judge-model "gpt-4.1-mini" \
    #     --qa-batch-size 1 \
    #     --judge-batch-size 1 \
    #     --api-config-path "/your/path/to/LightMem/src/lightmem/memory_toolkits/memories/configs/api_eval.json" \
    #     > "$log_file" 2>&1 &

    echo $! > "$pid_file"
    sleep 10
done