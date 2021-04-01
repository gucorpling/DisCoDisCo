#!/bin/bash
set -o errexit
if [ $# -eq 0 ]; then
	echo "Supply the name of a corpus"
	exit 1
fi
CORPUS="$1"
CORPUS_DIR="data/2019/${1}"
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
export EMBEDDING_DIMS=768
if [[ "$CORPUS" == "eng"* ]]; then 
	export EMBEDDING_MODEL_NAME="roberta-base"
	#export EMBEDDING_MODEL_NAME="bert-large-cased"
	#export EMBEDDING_DIMS=1024
elif [[ "$CORPUS" == "zho"* ]]; then
	export EMBEDDING_MODEL_NAME="bert-base-chinese"
	#export EMBEDDING_MODEL_NAME="hfl/chinese-roberta-wwm-ext-large"
	#export EMBEDDING_DIMS=1024
elif [[ "$CORPUS" == "deu"* ]]; then
  export EMBEDDING_MODEL_NAME="bert-base-german-cased"
elif [[ "$CORPUS" == "fra"* ]]; then
  export EMBEDDING_MODEL_NAME="camembert-base"
elif [[ "$CORPUS" == "nld"* ]]; then
	export EMBEDDING_MODEL_NAME="GroNLP/bert-base-dutch-cased"
elif [[ "$CORPUS" == "por"* ]]; then
  export EMBEDDING_MODEL_NAME="neuralmind/bert-base-portuguese-cased"
elif [[ "$CORPUS" == "rus"* ]]; then
  export EMBEDDING_MODEL_NAME="DeepPavlov/rubert-base-cased"
	#export EMBEDDING_MODEL_NAME="xlm-roberta-base"
elif [[ "$CORPUS" == "eus"* ]]; then
	export EMBEDDING_MODEL_NAME="ixa-ehu/berteus-base-cased"
# elif [[ "$CORPUS" == "spa"* ]]; then
# 	export EMBEDDING_MODEL_NAME="??"
elif [[ "$CORPUS" == "tur"* ]]; then
	export EMBEDDING_MODEL_NAME="dbmdz/bert-base-turkish-cased"
else
	#export EMBEDDING_MODEL_NAME="bert-base-multilingual-cased"
	export EMBEDDING_MODEL_NAME="xlm-roberta-base"
fi

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Training on $CORPUS"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
export TRAIN_DATA_PATH="${CORPUS_DIR}/${CORPUS}_train.conll"
export VALIDATION_DATA_PATH="${CORPUS_DIR}/${CORPUS}_dev.conll"
echo $TRAIN_DATA_PATH
allennlp train \
	configs/seg/baseline/bert_baseline.jsonnet \
	-s "$MODEL_DIR"
echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Testing on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
JSON_PRED_PATH="${MODEL_DIR}/output_test.jsonl"
CONLL_PRED_PATH="${MODEL_DIR}/output_test.conll"
CONLL_GOLD_PATH="${CORPUS_DIR}/${CORPUS}_test.conll"
allennlp predict \
	"${MODEL_DIR}/model.tar.gz" \
	"$CONLL_GOLD_PATH" \
	--use-dataset-reader \
	--cuda-device 0 \
	--output-file "$JSON_PRED_PATH"
echo "Removing model files..."
rm $MODEL_DIR/*.th
echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Scoring on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
python seg_scripts/format_output.py "$JSON_PRED_PATH" "$CONLL_PRED_PATH"
python seg_scripts/seg_eval_2019_modified.py "$CONLL_GOLD_PATH" "$CONLL_PRED_PATH" | tee "$MODEL_DIR/score.txt"
printf "#!/bin/sh\npython seg_scripts/seg_eval_2019_modified.py $CONLL_GOLD_PATH $CONLL_PRED_PATH\n" > "$MODEL_DIR/calc_score.sh"
