# Usage

## Setup
1. Create a new environment:

```bash
conda create --name disrpt python=3.8
conda activate disrpt
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the 2021 shared task data is at `data/2021/`.

## Experiments

Gold segmentation:

```bash
bash seg_scripts/single_corpus_train_and_test_ft.sh zho.rst.sctb
```

Silver segmentation:

```bash
bash seg_scripts/silver_single_corpus_train_and_test_ft.sh zho.rst.sctb
```

Relation classification:

```bash
bash rel_scripts/run_single_flair_clone.sh zho.rst.sctb
```

### Troubleshooting
Batch size may be modified, if necessary, using the `batch_size` parameter in:

* [`configs/seg/baseline/bert_baseline_ft.jsonnet`](configs/seg/baseline/bert_baseline_ft.jsonnet)
* [`configs/seg/baseline/bert_baseline_ft_silver.jsonnet`](configs/seg/baseline/bert_baseline_ft_silver.jsonnet)
* [`configs/rel/flair_clone.jsonnet`](configs/rel/flair_clone.jsonnet)
