#!/bin/bash

if [ $# -eq 0 ]; then
	echo "Supply the name of a corpus"
	exit 1
fi

CORPUS="$1"
CORPUS_DIR="sharedtask2021/data/${1}"
MODEL_DIR=${2:-models}/${CORPUS}_rel_bert_base

if [[ ! -d $CORPUS_DIR ]]; then
	echo "Corpus \"$CORPUS_DIR\" not found"
	exit 1
fi

if [[ -d $MODEL_DIR ]]; then
	echo "\"$MODEL_DIR\" already exists. Removing it now..."
	rm -rf "$MODEL_DIR"
fi

# use language-specific berts if we can
#export EMBEDDING_DIMS=768
if [[ "$CORPUS" == "eng"* ]]; then
	export EMBEDDING_MODEL_NAME="bert-base-cased"
elif [[ "$CORPUS" == "zho"* ]]; then
	export EMBEDDING_MODEL_NAME="bert-base-chinese"
elif [[ "$CORPUS" == "eus"* ]]; then
	export EMBEDDING_MODEL_NAME="ixa-ehu/berteus-base-cased"
elif [[ "$CORPUS" == "por"* ]]; then
  export EMBEDDING_MODEL_NAME="neuralmind/bert-base-portuguese-cased"
elif [[ "$CORPUS" == "tur"* ]]; then
	export EMBEDDING_MODEL_NAME="dbmdz/bert-base-turkish-cased"
else
	export EMBEDDING_MODEL_NAME="bert-base-multilingual-cased"
fi
#export EMBEDDING_MODEL_NAME="distilbert-base-multilingual-cased"

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Training on $CORPUS"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
export TRAIN_DATA_PATH="${CORPUS_DIR}/${CORPUS}_train.rels"
export VALIDATION_DATA_PATH="${CORPUS_DIR}/${CORPUS}_dev.rels"
echo $TRAIN_DATA_PATH
allennlp train \
	configs/rel/e2e/e2e.jsonnet \
	-s "$MODEL_DIR" \
#	-o overrides,