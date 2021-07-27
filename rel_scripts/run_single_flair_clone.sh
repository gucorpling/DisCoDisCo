#!/bin/bash
if [ $# -eq 0 ]; then
  echo "Supply the name of a corpus"
  exit 1
fi
if [[ ! -d "data/" ]]; then
        echo "Data not found--please download it from https://drive.google.com/file/d/1wDmv6TzZqUwnw1Csn4Yz66uF1UYV-K-L/view?usp=sharing"
        exit 1
fi

CORPUS="$1"
CORPUS_DIR="data/2021/${1}"
MODEL_DIR=${2:-models}/${CORPUS}_flair_clone

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
if [[ "$CORPUS" == "eng"* ]]; then
  export EMBEDDING_DIMS=1024
  #export EMBEDDING_MODEL_NAME="roberta-large"
  export EMBEDDING_MODEL_NAME="google/electra-large-discriminator"
elif [[ "$CORPUS" == "fas"* ]]; then
  export EMBEDDING_MODEL_NAME="HooshvareLab/bert-fa-base-uncased"
elif [[ "$CORPUS" == "deu"* ]]; then
  #export EMBEDDING_DIMS=1024
  #export EMBEDDING_MODEL_NAME="deepset/gelectra-large"
  export EMBEDDING_DIMS=1024
  export EMBEDDING_MODEL_NAME="deepset/gbert-large"
elif [[ "$CORPUS" == "fra"* ]]; then
  export EMBEDDING_MODEL_NAME="dbmdz/bert-base-french-europeana-cased"
elif [[ "$CORPUS" == "zho"* ]]; then
  #export EMBEDDING_DIMS=1024
  #export EMBEDDING_MODEL_NAME="hfl/chinese-electra-180g-large-discriminator"
  export EMBEDDING_MODEL_NAME="bert-base-chinese"
elif [[ "$CORPUS" == "nld"* ]]; then
  #export EMBEDDING_MODEL_NAME="GroNLP/bert-base-dutch-cased"
  export EMBEDDING_MODEL_NAME="pdelobelle/robbert-v2-dutch-base"
elif [[ "$CORPUS" == "eus"* ]]; then
  export EMBEDDING_MODEL_NAME="ixa-ehu/berteus-base-cased"
elif [[ "$CORPUS" == "spa"* ]]; then
  #export EMBEDDING_MODEL_NAME="mrm8488/electricidad-base-discriminator"
  export EMBEDDING_MODEL_NAME="dccuchile/bert-base-spanish-wwm-cased"
elif [[ "$CORPUS" == "por"* ]]; then
  export EMBEDDING_MODEL_NAME="neuralmind/bert-base-portuguese-cased"
elif [[ "$CORPUS" == "tur"* ]]; then
  #export EMBEDDING_MODEL_NAME="dbmdz/electra-base-turkish-cased-discriminator"
  export EMBEDDING_MODEL_NAME="dbmdz/bert-base-turkish-cased"
elif [[ "$CORPUS" == "rus"* ]]; then
  export EMBEDDING_MODEL_NAME="DeepPavlov/rubert-base-cased"
else
  export EMBEDDING_DIMS=1024
  export EMBEDDING_MODEL_NAME="xlm-roberta-large"
fi

# use fastText embeddings
if [[ "$CORPUS" == "eng"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.en.300.vec"
elif [[ "$CORPUS" == "deu"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.de.300.vec"
elif [[ "$CORPUS" == "eus"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.eu.300.vec"
elif [[ "$CORPUS" == "fas"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.fa.300.vec"
elif [[ "$CORPUS" == "fra"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.fr.300.vec"
elif [[ "$CORPUS" == "nld"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.nl.300.vec"
elif [[ "$CORPUS" == "por"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.pt.300.vec"
elif [[ "$CORPUS" == "rus"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.ru.300.vec"
elif [[ "$CORPUS" == "spa"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.es.300.vec"
elif [[ "$CORPUS" == "tur"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.nl.300.vec"
elif [[ "$CORPUS" == "zho"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.zh.300.vec"
else
  echo "Couldn't find a fasttext embedding for \"$CORPUS\"" >&2
  exit 1
fi


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
        --use-dataset-reader \
        --output-file $OUTPUT_FILE_PATH

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Evaluating on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
python utils/e2e_metrics.py $OUTPUT_FILE_PATH
cat $MODEL_DIR/predicti.res
