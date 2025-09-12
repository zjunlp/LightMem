mkdir -p logs

ranges=(
    "0 100"
    "100 200"
    "200 300"
    "300 400"
    "400 500"
)

api_keys=(
    "sk-UeZijRSOLATIQ3ztg8gXFpIMCq2AEEugw1kqHHREj2eKoUPc"
    "sk-Oly75BVdkKVKpc4vmczhDMlX2DzxhUvlmyhCOSxZw2R4qxg1"
    "sk-npZ3sLDRTI26hRnN7UYrD7UHj9JmC0lBHFNdqUn8IU4iImUY"
    "sk-UclXmbx2IxrqR7C6YQRVZtwbJs1YEaY9AGMAfUmfTkIjrRAG"
    "sk-z78AerrKKzgF1hEeCcwbu9KRC23kwU6Uxy6Pz4JM3McUZqPJ"
)

base_urls=(
    "https://api.gpts.vin/v1"
    "https://api.gpts.vin/v1"
    "https://api.gpts.vin/v1"
    "https://api.gpts.vin/v1"
    "https://api.gpts.vin/v1"
)

for i in {1..1}; do
    read start_idx end_idx <<< "${ranges[$i]}"
    export OPENAI_API_KEY="${api_keys[$i]}" 
    export OPENAI_API_BASE="${base_urls[$i]}"
    nohup python memory_construction.py \
        --memory-type "LangMem" \
        --dataset-type "LongMemEval" \
        --dataset-path "/mnt/dengxinle/raspberry/memory_benchmark/longmemeval/longmemeval_s.json" \
        --config-path "langmem_config.json" \
        --num-workers 4 \
        --start-idx "$start_idx" \
        --end-idx "$end_idx" \
        --token-cost-save-filename "token_cost_$((i+1))_${start_idx}_${end_idx}" \
        --rerun \
        > "logs/process_$((i+1))_${start_idx}_${end_idx}.log" 2>&1 &

    echo $! > "logs/process_$((i+1)).pid"
    sleep 10
done