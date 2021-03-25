#!/bin/bash
set -o errexit
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
allennlp predict \
	"${MODEL_DIR}/model.tar.gz" \
	"sharedtask2019/data/${CORPUS}/${CORPUS}_test.conll" \
	--use-dataset-reader \
	--cuda-device 0 \
	--output-file "${MODEL_DIR}/output_test.jsonl"
echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "#### Scoring on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
python seg_scripts/score_binary_output.py "${MODEL_DIR}/output_test.jsonl" > "${MODEL_DIR}/metrics_test.txt"
cat "${MODEL_DIR}/metrics_test.txt"
echo "\n(Scores saved in ${MODEL_DIR}/metrics_test.txt)"
