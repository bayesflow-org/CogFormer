# CogFormer
Learn All Your Models Once

![CogFormer Architecture](cogformer/architecture.png)

Simulation-based inference (SBI) with neural networks has accelerated and transformed cognitive modeling workflows.
SBI enables modelers to fit complex models that were previously difficult or impossible to estimate, while also allowing rapid estimation across large numbers of datasets.
However, the utility of SBI for iterating over varying modeling assumptions remains limited: changing parameterizations, generative functions, priors, and design variables all necessitate model retraining and hence diminish the benefits of amortization.
To address these issues, we pilot a meta-amortized framework for cognitive modeling which we nickname the CogFormer.
Our framework trains a transformer-based architecture that remains valid across a combinatorial number of structurally similar models, allowing for changing data types, parameters, design matrices, and sample sizes.
We present promising quantitative results across families of decision-making models for binary, multi-alternative, and continuous responses. Our evaluation suggests that CogFormer can accurately estimate parameters across model families with a minimal amortization offset, making it a potentially powerful engine that catalyzes cognitive modeling workflows.

## Installation

Requires Python 3.12+. Clone the repo and install with [uv](https://github.com/astral-sh/uv) (recommended):

```bash
git clone https://github.com/jerrymhuang/CogFormer.git
cd CogFormer
uv sync
```

Or with pip:

```bash
git clone https://github.com/jerrymhuang/CogFormer.git
cd CogFormer
pip install -e .
```

## Citation
```text
@article{huang2026cogformer,
  title={CogFormer: Learn All Your Models Once},
  author={Huang, Jerry M and Schumacher, Lukas and Stevenson, Niek and Radev, Stefan T},
  journal={arXiv preprint arXiv:2603.20520},
  year={2026}
}
```

## Artifacts and checkpoints

The repository tracks **code and small summary tables** (`outputs/tables/`, `outputs/docs/`)
so results are reproducible from the scripts in `experiments/`. Generated **figures**
(`outputs/figures/*.pdf`, `*.png`) and **model checkpoints** (`outputs/checkpoints/*.pt`)
are *not* tracked in git — regenerate figures by re-running the experiment scripts.

Trained checkpoints are hosted on the Hugging Face Hub:
`<HF_REPO_ID>` (TODO: publish). Load them at runtime with:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(repo_id="<HF_REPO_ID>", filename="cogformer_mixed_attn_l8_h8_p256_s32_d64_o500_b64_e5000_t100.pt")
```