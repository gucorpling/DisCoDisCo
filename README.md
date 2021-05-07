# Usage

## Setup
1. Data is included by linking to the CODI-DISRPT 2021 repo as a submodule. Make sure you've cloned it:

```bash
git submodule update --init
```

2. Create a new environment:

```bash
conda create --name disrpt python=3.8
conda activate disrpt
```

3. (*optional*) If you have a GPU, set up CUDA and PyTorch before other dependencies. 
   (You may need to modify the version of `cudatoolkit` based on your driver versions. 
   If you're on macOS or Linux, you can type `nvidia-smi` to see your driver version.
   See: https://docs.nvidia.com/deploy/cuda-compatibility/index.html)

```bash
conda install pytorch torchvision cudatoolkit=11 -c pytorch
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Experiments

Instead of running Python code, you'll be invoking the `allennlp` command in your shell using one of the configs
in [`configs/`](configs/). Configs are designed to help you capture every detail of an experiment in a single file,
which helps us replicate our results. Moreover, it makes it really easy for us to swap out components, e.g. 
one flavor of BERT for another.

### Training

Make sure you've activated your environment:

```bash
conda activate disrpt
```

Invoke `allennlp train`. The first argument tells AllenNLP which configuration to use, and the `-s` flag tells
it where to save the model that it trains.

```bash
allennlp train configs/rel/baseline/distilbert_baseline.jsonnet -s models/rel_distilbert_baseline
# or 
allennlp train configs/seg/baseline/distilbert_baseline.jsonnet -s models/seg_distilbert_baseline
```

### Predicting

Invoke `allennlp predict`. The first argument tells it which model to use, and the second argument tells it 
which input data to predict on.

```bash
# note: if you have a CUDA-capable GPU, also pass `--cuda-device 0`
allennlp predict \ 
        models/distilbert_baseline/model.tar.gz \
        sharedtask2021/data/eng.rst.gum/eng.rst.gum_dev.rels \
        --use-dataset-reader \
        --output-file /tmp/predictions.txt
# or 
allennlp predict \
        models/seg_distilbert_baseline/model.tar.gz \
        sharedtask2019/data/eng.rst.gum/eng.rst.gum_test.conll \
        --use-dataset-reader \
        --output-file /tmp/predictions.txt
```

Output is formatted as a series of JSON lines. For example:

```bash
# the instance we're predicting on
input 162:  Instance with fields:
 	 unit1_body: TextField of length 5 with text: 
 		[[CLS], to, make, sure, [SEP]]
 		and TokenIndexers : {'tokens': 'PretrainedTransformerIndexer'} 
 	 unit1_sentence: TextField of length 52 with text: 
 		[[CLS], With, these, in, hand, ,, we, will, design, the, curriculum, of, clinics, ,, based, on, the,
		method, of, ', construct, ##ive, alignment, ', (, Big, ##gs, et, al, ., ,, 2011, ), ,, to, make,
		sure, that, the, intended, learning, objectives, and, the, teaching, /, learning, activities, stay,
		aligned, ., [SEP]]
 		and TokenIndexers : {'tokens': 'PretrainedTransformerIndexer'} 
 	 unit2_body: TextField of length 16 with text: 
 		[[CLS], that, the, intended, learning, objectives, and, the, teaching, /, learning, activities,
		stay, aligned, ., [SEP]]
 		and TokenIndexers : {'tokens': 'PretrainedTransformerIndexer'} 
 	 unit2_sentence: TextField of length 52 with text: 
 		[[CLS], With, these, in, hand, ,, we, will, design, the, curriculum, of, clinics, ,, based, on, the,
		method, of, ', construct, ##ive, alignment, ', (, Big, ##gs, et, al, ., ,, 2011, ), ,, to, make,
		sure, that, the, intended, learning, objectives, and, the, teaching, /, learning, activities, stay,
		aligned, ., [SEP]]
 		and TokenIndexers : {'tokens': 'PretrainedTransformerIndexer'} 
 	 relation: LabelField with label: attribution in namespace: 'relation_labels'. 
 	 direction: LabelField with label: 1>2 in namespace: 'direction_labels'. 

# the prediction output
{
   "pred_relation":"attribution",
   "gold_relation":"attribution",
   "loss":1.1672526597976685,
      "relation_probs":{
      "elaboration":0.0178113654255867,
      "joint":0.0076534380204975605,
      "sequence":0.012619145214557648,
      "attribution":0.8847652673721313,
      "background":0.003287180792540312,
      "evidence":0.021079568192362785,
      "preparation":1.6962607332970947e-05,
      "circumstance":0.0327010415494442,
      "evaluation":6.268547440413386e-05,
      "contrast":7.695440217503347e-06,
      "concession":0.007866458036005497,
      "restatement":0.004466917831450701,
      "purpose":0.0014542682329192758,
      "result":0.005173283629119396,
      "cause":0.0007559513323940337,
      "justify":0.00010502728400751948,
      "manner":0.00011868654837599024,
      "condition":2.267566287628142e-06,
      "antithesis":3.0185639843693934e-05,
      "question":1.4590353925036936e-10,
      "means":1.403105488861911e-05,
      "motivation":8.482756129524205e-06,
      "solutionhood":4.333505643216995e-08
   }
}
```

### Ensemble Segmentation Module

Each of the three modules will produce JSON output in `models/seg_ensemble_jsons/{allennlp,flair,subtree}/{zho.rst.sctb,...}.json`.
To run the ensemble modules for a single corpus, invoke the following command:

```bash
bash seg_scripts/ensemble/single_corpus_train_and_test.sh zho.rst.sctb
```

To clean up after a run, just `rm -rf models/`.

### Setting up configs

TODO, but see https://guide.allennlp.org/using-config-files/

### Coding New Models

TODO, but for now see [baseline_model](gucorpling_models/rel/baseline_model.py)


# AllenNLP Template Project using Config Files

A template for starting a new allennlp project using config files and `allennlp train`.  For simple
projects, all you need to do is get your model code into the class in `my_project/model.py`, your
data loading code into the `DatasetReader` code in `my_project/dataset_reader.py`, and that's it,
you can train a model (we recommend also making appropriate changes to the test code, and using that
for development, but that's optional).

See the [AllenNLP Guide](https://guide.allennlp.org/your-first-model) for a quick start on how to
use what's in this example project.  We're grabbing the model and dataset reader classes from that
guide.  You can replace those classes with a model and dataset reader for whatever you want
(including copying code from our [model library](https://github.com/allenai/allennlp-models) as a
starting point). The very brief version of what's in here:

* A `Model` class in `my_project/model.py`.
* A `DatasetReader` class in `my_project/dataset_reader.py`.
* Tests for both of these classes in `tests`, including a small toy dataset that can be read.  We
  strongly recommend that you use a toy dataset with tests like this during model development, for
  quick debug cycles. To run the tests, just run `pytest` from the base directory of this project.
* An `.allennlp_plugins` file, which makes it easier to use `allennlp train` with this project.  If
  you change the name of `my_project`, you should also change the line in this file to match.
* An example configuration file for training the model, which you would use with the following shell
  command from the base directory of this repository, after `pip install allennlp`:

  `allennlp train -s /dir/to/save/results training_config/my_model_trained_on_my_dataset.json`
