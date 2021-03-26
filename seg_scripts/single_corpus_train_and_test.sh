#!/bin/bash
if [ $# -eq 0 ]; then
	echo "Supply the name of a corpus"
	exit 1
fi
CORPUS="$1"
CORPUS_DIR="sharedtask2019/data/${1}"
MODEL_DIR=models/${CORPUS}_seg_bert_baseline
if [[ ! -d $CORPUS_DIR ]]; then
	echo "Corpus \"$CORPUS_DIR\" not found"
	exit 1
fi
if [[ -d $MODEL_DIR ]]; then
	echo "\"$MODEL_DIR\" already exists. Please remove it."
	exit 1
fi


# use language-specific berts if we can
if [[ "$CORPUS" == "eng"* ]]; then 
	export EMBEDDING_MODEL_NAME="bert-base-cased"
elif [[ "$CORPUS" == "zho"* ]]; then
	export EMBEDDING_MODEL_NAME="bert-base-chinese"
else
	export EMBEDDING_MODEL_NAME="bert-base-multilingual-cased"
fi

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "#### Training on $CORPUS"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
export TRAIN_DATA_PATH="sharedtask2019/data/${CORPUS}/${CORPUS}_train.conll"
export VALIDATION_DATA_PATH="sharedtask2019/data/${CORPUS}/${CORPUS}_dev.conll"
echo $TRAIN_DATA_PATH
allennlp train \
	configs/seg/baseline/bert_baseline.jsonnet \
	-s "$MODEL_DIR"
echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "#### Testing on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
JSON_PRED_PATH="${MODEL_DIR}/output_test.jsonl"
CONLL_PRED_PATH="${MODEL_DIR}/output_test.conll"
CONLL_GOLD_PATH="sharedtask2019/data/${CORPUS}/${CORPUS}_test.conll"
allennlp predict \
	"${MODEL_DIR}/model.tar.gz" \
	"$CONLL_GOLD_PATH" \
	--use-dataset-reader \
	--cuda-device 0 \
	--output-file "$JSON_PRED_PATH"
set -o errexit
echo "Removing model files..."
rm $MODEL_DIR/*.th
echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "#### Scoring on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
python seg_scripts/format_output.py "$JSON_PRED_PATH" "$CONLL_PRED_PATH"
python sharedtask2019/utils/seg_eval.py "$CONLL_GOLD_PATH" "$CONLL_PRED_PATH"
