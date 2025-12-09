# Baseline Pipeline README

This repository provides a simple script to run the full baseline pipeline (memory construction → memory search → memory evaluation). Follow the steps below to run everything smoothly.

---

## 1. Configure the Script

Open the bash script and fill in the following variables:

* `dataset_path`: path to your dataset
* `config_path`: path to your memory configuration
* `api_keys`: your API keys
* `base_urls`: corresponding base URLs

```bash
api_keys=(
    "YOUR_API_KEY_1"
    "YOUR_API_KEY_2"
    ...
)

base_urls=(
    "YOUR_BASE_URL_1"
    "YOUR_BASE_URL_2"
    ...
)
```

You may also modify the data ranges if needed:

```bash
ranges=(
    "0 100"
    "100 200"
    ...
)
```

---

## 2. Run Stage 1: Memory Construction

Set the Python script name inside the bash file to:

```bash
memory_construction.py
```

Then launch the pipeline:

```bash
bash run_baseline.sh
```

Wait until all processes finish.

---

## 3. Run Stage 2: Memory Search

After Stage 1 is finished, change the Python script inside the bash file to:

```bash
memory_search.py
```

Run again:

```bash
bash run_baseline.sh
```

---

## 4. Run Stage 3: Memory Evaluation

Finally, change the script to:

```bash
memory_evaluation.py
```

Run the bash script once more:

```bash
bash run_baseline.sh
```

---

## 5. Final Outputs

After all three stages are completed, you will obtain:

* memory construction logs
* memory search results
* evaluation metrics

---

This is the minimal workflow needed to run the baseline end-to-end.
