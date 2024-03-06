CORPUS="eng.rst.gum"
export CORPUS
CORPUS_DIR="data/erst/${1}"
MODEL_DIR=${2:-models}/erst_${1}_flair_clone

if [[ ! -d $CORPUS_DIR ]]; then
  echo "Corpus \"$CORPUS_DIR\" not found"
  exit 1
fi

if [[ -d $MODEL_DIR ]]; then
   # echo "\"$MODEL_DIR\" already exists. Ignore it"
   # exit 1
  echo "\"$MODEL_DIR\" already exists. Removing it now..."
  rm -rf "$MODEL_DIR"
fi

# use language-specific berts if we can
export EMBEDDING_DIMS=768
export EMBEDDING_MODEL_NAME="bert-base-cased"

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Training on $CORPUS"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
export TRAIN_DATA_PATH="${CORPUS_DIR}/${CORPUS}_train.rels"
export VALIDATION_DATA_PATH="${CORPUS_DIR}/${CORPUS}_dev.rels"
echo $TRAIN_DATA_PATH
allennlp train \
  configs/rel/flair_clone.jsonnet \
  -s "$MODEL_DIR" \

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Predicting on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
export TEST_DATA_PATH="${CORPUS_DIR}/${CORPUS}_test.rels"
export OUTPUT_FILE_PATH="$MODEL_DIR/test_predictions.json"
echo $TEST_DATA_PATH
allennlp predict \
        $MODEL_DIR \
        $TEST_DATA_PATH \
        --silent \
	--cuda-device 0 \
        --use-dataset-reader \
        --output-file $OUTPUT_FILE_PATH

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Evaluating on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
python utils/e2e_metrics.py $OUTPUT_FILE_PATH
cat $MODEL_DIR/predicti.res
