<div align="center">
  <h1>StructMem</h1>
  <p><b>Structured memory for long-horizon behavior in language and multimodal models.</b></p>
</div>

<p align="center">
  <a href="https://github.com/zjunlp/LightMem/tree/StructMem"><img src="https://img.shields.io/badge/Project-StructMem-4c6fff" alt="StructMem project"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>

<p align="center">
  <a href="#quick-start"><b>🚀 Quick Start</b></a> &nbsp;·&nbsp;
  <a href="#how-it-works"><b>🏗️ How It Works</b></a> &nbsp;·&nbsp;
  <a href="#repository-layout"><b>📦 Repository</b></a> &nbsp;·&nbsp;
  <a href="#citation"><b>📄 Citation</b></a>
</p>

<div align="center">
  <img src="text/figs/StructMem.png" width="92%" alt="StructMem overview">
</div>

<details>
<summary><b>📑 Table of contents</b></summary>

- [📢 News](#news)
- [🎯 What is StructMem](#what-is-structmem)
- [🚀 Quick Start](#quick-start)
- [🏗️ How It Works](#how-it-works)
- [📦 Repository Layout](#repository-layout)
- [🧪 Tests and Evaluation](#tests-and-evaluation)
- [📄 Citation](#citation)
- [⚖️ License](#license)

</details>

---

<span id="news"></span>

## 📢 News

- **[2026-10-26]**: 🗂️ A standalone StructMem repository was created as the project grew beyond text-only memory; this branch now hosts the text and multimodal implementations.
- **[2026-10-24]**: 🖼️ StructMem now supports **multimodal memory**.
- **[2026-04-24]**: 🎉🎉🎉 [**StructMem: Structured Memory for Long-Horizon Behavior in LLMs**](https://arxiv.org/abs/2604.21748) has been accepted by **ACL 2026**!

<span id="what-is-structmem"></span>

## 🎯 What is StructMem

StructMem builds memories around events instead of isolated flat facts.

| Implementation | Best for | Entry point |
| --- | --- | --- |
| 📝 **Text** | Event-level factual and relational extraction with temporal anchoring and cross-event summarization. | [`text/`](text/README.md) |
| 🖼️ **Multimodal** | The same direction to ordered text and image evidence with persistent, user-isolated retrieval. | [`multimodal/`](multimodal/README.md) |

<span id="quick-start"></span>

## 🚀 Quick Start

Clone the StructMem branch first:

```bash
git clone --branch StructMem --single-branch https://github.com/zjunlp/LightMem.git
cd LightMem
```

### 📝 Text implementation

```bash
cd text
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The canonical Python package namespace is `structmem`:

```python
from structmem.memory.structmem import StructMemory
```

### 🖼️ Multimodal service

```bash
cd multimodal
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export MEMORY_API_KEY='change-me'
python -m uvicorn service.multimodal:app --host 127.0.0.1 --port 18084
```

<span id="how-it-works"></span>

## 🏗️ How It Works

### Text memory

1. **Event extraction** preserves factual components, relational components, and temporal bindings.
2. **Hierarchical consolidation** connects related events within a time window and stores cross-event summaries alongside detailed memories.
3. **Scoped retrieval** combines the resulting memories for long-horizon question answering.

### Multimodal memory

1. **Ingestion** stores original text and image evidence with ordered metadata.
2. **Encoding** supports remote multimodal embeddings, optional local CLIP, and lexical matching.
3. **Evidence retrieval** combines the available signals and returns source evidence with user isolation and persistent SQLite storage.

<span id="repository-layout"></span>

## 📦 Repository Layout

```text
StructMem/
├── text/                 # Original text-only StructMem implementation
│   ├── src/structmem/    # Canonical Python package namespace
│   ├── experiments/      # LoCoMo construction and evaluation scripts
│   └── StructMem.md      # Method guide and examples
├── multimodal/           # Standalone multimodal memory service
│   ├── service/           # FastAPI application and retrieval modules
│   ├── tests/             # Synthetic, credential-free tests
│   └── SERVICE.md         # API limits and configuration details
└── README.md
```

Datasets, credentials, model weights, runtime databases, generated memories, and benchmark outputs stay outside the repository.

<span id="tests-and-evaluation"></span>

## 🧪 Tests and Evaluation

Run the multimodal unit tests with synthetic content and mocked encoders:

```bash
cd multimodal
python -m pytest -q
```

The text project contains the LoCoMo construction and search scripts.
Their dataset paths, model endpoints, and output directories are supplied by the user; see [`text/experiments/locomo/readme.md`](text/experiments/locomo/readme.md).
Published or development scores should always include the exact dataset split, model, evaluator, and retrieval configuration.

<span id="citation"></span>

## 📄 Citation

Please cite the StructMem paper if you use the code or ideas in your work:

```bibtex
@inproceedings{xu2026structmem,
  author       = {Xu, Buqiang and Chen, Yijun and Fang, Jizhan and Zhong, Ruobin and Yao, Yunzhi and Zhu, Yuqi and Du, Lun and Deng, Shumin},
  title        = {StructMem: Structured Memory for Long-Horizon Behavior in LLMs},
  booktitle    = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), {ACL} 2026, San Diego, California, United States, July 2-7, 2026},
  pages        = {122--146},
  publisher    = {Association for Computational Linguistics},
  year         = {2026},
  url          = {https://doi.org/10.18653/v1/2026.acl-short.12},
  doi          = {10.18653/V1/2026.ACL-SHORT.12}
}
```


<span id="license"></span>

## ⚖️ License

The implementations are released under the MIT License.
