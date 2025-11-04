# Evaluating Long-Term Memory for LLMs with LightMem

This project demonstrates a robust system for endowing Large Language Models (LLMs) with long-term memory capabilities using the **LightMem** framework. By integrating a locally hosted Qwen model via an OpenAI-compatible API, this experiment showcases how to overcome the inherent limitations of finite context windows, enabling conversations that are context-aware, persistent, and personalized across multiple sessions.

The included script runs a series of tests to evaluate the system's performance on three critical long-term memory tasks:
1.  **Multi-Session Context Consolidation**
2.  **Dynamic Knowledge Updates**
3.  **Handling of Unanswerable Questions**

## 🏛️ Architecture

The system operates on a Retrieval-Augmented Generation (RAG) architecture orchestrated by LightMem:

-   **Memory Framework:** **LightMem** manages the entire lifecycle of memories: ingestion, compression, storage, retrieval, and updating.
-   **Reasoning Engine (LLM):** **Qwen/Qwen3-Next-80B-A3B-Instruct** (or any other compatible model) served locally, responsible for understanding user queries and generating responses based on retrieved memories.
-   **Memory Compressor:** **`llmlingua-2`** reduces the token footprint of conversational history, making storage and retrieval more efficient.
-   **Embedding Model:** **`all-MiniLM-L6-v2`** transforms textual memories into dense vector representations for semantic search.
-   **Vector Store:** **Qdrant** provides a high-performance local database for storing and querying memory vectors.

## ✨ Key Features Demonstrated

-   **Cross-Session Information Recall:** The system can synthesize information provided in separate conversations to answer a complex query.
-   **Stateful Knowledge Management:** It correctly identifies and uses the most recent information when user preferences or facts change over time.
-   **Knowledge Boundary Awareness:** The model can recognize when it lacks the necessary information to answer a question and gracefully abstains from responding, preventing factual hallucination.

## 🚀 Getting Started

Follow these steps to set up and run the experiment on your own machine.

### 1. Prerequisites

-   Python 3.8+
-   An OpenAI-compatible API server running a model like Qwen. You can set this up using tools like [vLLM](https://github.com/vllm-project/vllm) or [FastChat](https://github.com/lm-sys/FastChat).
-   The required Hugging Face models downloaded locally.

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/zjunlp/LightMem.git
    cd LightMem
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    # Using conda
    conda create -n lightmem python=3.10
    conda activate lightmem
    
    # Or using venv
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If you don't have a `requirements.txt` file, create one with the following content)*:
    
    ```
    lightmem
    openai
    tqdm
    ```

### 3. Configuration

Before running the script, you **must** update the configuration variables at the top of `experiments/run_lightmen_qwen_server.py`:

```python
# ============ API Configuration ============
# ⚠️ UPDATE THIS to your LLM server's endpoint
API_BASE_URL = "your_qwen_server_url"

# ============ Model Paths ============
# ⚠️ UPDATE THESE paths to where you have downloaded the models
LLMLINGUA_MODEL_PATH = "/your/path/to/models/llmlingua-2-bert-base-multilingual-cased-meetingbank"
EMBEDDING_MODEL_PATH = "/your/path/to/models/all-MiniLM-L6-v2"

# ============ Data Configuration ============
# This points to the sample data file
DATA_PATH = "/your/path/to/LightMem/experiments/sample_data.json"
```

### 4. Running the Experiment

Navigate to the directory containing the script and execute it:

```bash
# It is recommended to run from the root directory of the project
python experiments/run_lightmen_qwen_server.py
```

## 📊 Expected Output

As the script runs, you will see the following:

1.  **Console Output:** A `tqdm` progress bar will show the progress through the three test cases. The generated answers from the LLM and the "yes/no" judgment will be printed to the console for each case.

    ```
    0%|                                                              | 0/3 [00:00<?, ?it/s]
    multi_session_trip_planning_01
    ...
    Based on the memories provided: ... You decided on Tokyo ... in October.
    yes
    33%|█████████▋                                         | 1/3 [00:16<00:32, 16.45s/it]
    ...
    100%|████████████████████████████████████████████| 3/3 [00:23<00:00,  7.94s/it]
    ```

2.  **Generated Files:** Two new directories will be created in your project root:
    
    -   `./qdrant_data/`: Contains the local Qdrant database files where the vectorized memories are stored for each test case.
    -   `./results/`: Contains detailed JSON output files for each test case (e.g., `result_multi_session_trip_planning_01.json`), logging the prompts, generated answers, ground truth, and correctness score.

## 🔬 Understanding the Test Cases

The evaluation is performed using the `sample_data.json` file, which contains three distinct scenarios.

### Test Case 1: `multi_session_trip_planning_01`

-   **Scenario:** A user first expresses interest in a trip to a big city, and in a later session, decides on Tokyo and the month of October.
-   **Objective:** To test if the system can retrieve and combine facts from different conversational sessions.
-   **Outcome:** The system correctly answers that the user decided on **Tokyo** for a trip in **October**, demonstrating successful context consolidation.

### Test Case 2: `knowledge_update_favorite_color_02`

-   **Scenario:** A user first states their favorite color is "blue," but in a later session, updates it to "green."
-   **Objective:** To test the system's ability to handle conflicting information and prioritize the most recent update.
-   **Outcome:** The system correctly identifies the user's current favorite color as **green**, proving its capability to manage dynamic, evolving knowledge.

### Test Case 3: `multi_session_trip_planning_abs_01`

-   **Scenario:** Following the trip planning conversations, the user asks for the name of the hotel they booked. This information was never mentioned.
-   **Objective:** To test if the system can recognize the absence of information in its memory and avoid making things up (hallucinating).
-   **Outcome:** The system correctly states that **no information about a hotel booking is available**, demonstrating robust knowledge boundary detection.