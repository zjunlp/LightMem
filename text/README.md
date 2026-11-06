# StructMem Text

The original text-only StructMem implementation for long-horizon conversation memory.
It preserves event-level bindings, temporal structure, and cross-event summaries in the canonical `structmem` Python namespace.

<p>
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white" alt="Python 3.10 or 3.11">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>

## ✨ What is included

- Event-level factual and relational extraction.
- Temporal anchoring for ordered conversation events.
- Cross-event summarization over configurable time windows.
- LoCoMo construction, retrieval, and evaluation scripts.

The method guide and examples are in [`StructMem.md`](StructMem.md).
The evaluation workflow is in [`experiments/locomo/readme.md`](experiments/locomo/readme.md).

## 🚀 Install

Use Python 3.10 or 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The package exposes the StructMem API:

```python
from structmem.memory.structmem import StructMemory
```

Select StructMem event extraction with `extraction_mode="event"` and configure the summary retriever as shown in [`StructMem.md`](StructMem.md).

## 🔬 Reproduce experiments

The LoCoMo scripts accept dataset paths, model endpoints, and output locations from the local experiment configuration.
Datasets, credentials, generated memories, and model outputs are intentionally not included here.

```bash
cd experiments/locomo
python add_locomo.py --help
python search_locomo.py --help
```

## Attribution and license

This code is the text implementation of StructMem, historically developed on top of the LightMem codebase and now maintained under the `structmem` package namespace.
Released under the MIT License; see [`LICENSE`](LICENSE).
