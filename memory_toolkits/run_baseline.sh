# An example of running the baseline model. 
# Please modify the following variables to fit your own dataset and memory configuration. 
# ========================================================
memory_type="LangMem"
dataset_type="LongMemEval"
dataset_path="YOUR_DATASET_PATH"
config_path="YOUR_MEMORY_CONFIG_PATH"
num_workers=4
tokenizer_path="gpt-4"
log_dir="langmem_logs"
token_cost_prefix="token_cost"
pid_prefix="process"
ranges=(
    "0 100"
    "100 200"
    "200 300"
    "300 400"
    "400 500"
)
api_keys=(
    "YOUR_API_KEY_1"
    "YOUR_API_KEY_2"
    "YOUR_API_KEY_3"
    "YOUR_API_KEY_4"
    "YOUR_API_KEY_5"
)
base_urls=(
    "YOUR_BASE_URL_1"
    "YOUR_BASE_URL_2"
    "YOUR_BASE_URL_3"
    "YOUR_BASE_URL_4"
    "YOUR_BASE_URL_5"
)
# ========================================================

[ ! -d "$log_dir" ] && mkdir -p "$log_dir"

for ((i=0; i<${#ranges[@]}; i++)); do
    read start_idx end_idx <<< "${ranges[$i]}"
    export OPENAI_API_KEY="${api_keys[$i]}" 
    export OPENAI_API_BASE="${base_urls[$i]}"

    log_file="${log_dir}/${pid_prefix}_$((i+1))_${start_idx}_${end_idx}.log"
    token_cost_file="${token_cost_prefix}_${memory_type,,}_$((i+1))_${start_idx}_${end_idx}"
    pid_file="${log_dir}/${pid_prefix}_$((i+1)).pid"

    [ ! -f "$log_file" ] && touch "$log_file"

    nohup python memory_construction.py \
        --memory-type "$memory_type" \
        --dataset-type "$dataset_type" \
        --dataset-path "$dataset_path" \
        --config-path "$config_path" \
        --num-workers "$num_workers" \
        --start-idx "$start_idx" \
        --end-idx "$end_idx" \
        --token-cost-save-filename "$token_cost_file" \
        --tokenizer-path "$tokenizer_path" \
        > "$log_file" 2>&1 &
    echo $! > "$pid_file"
    sleep 10
done