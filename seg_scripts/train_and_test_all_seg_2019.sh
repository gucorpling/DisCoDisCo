#!/bin/bash
set -o errexit
for CORPUS in `ls sharedtask2019/data`; do
        MODEL_DIR=models/${CORPUS}_seg_bert_baseline
	echo ""
	echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        echo "#### Training on ${CORPUS}"
        echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
	echo ""
	TRAIN_DATA_PATH="sharedtask2019/data/${CORPUS}/${CORPUS}_train.conll" \
		VALIDATION_DATA_PATH="sharedtask2019/data/${CORPUS}/${CORPUS}_dev.conll" \
		allennlp train \
		configs/seg/baseline/bert_baseline.jsonnet \
		-s ${MODEL_DIR}
	echo ""
        echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        echo "#### Testing on ${CORPUS}"
        echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        echo ""
        allennlp predict \
                --cuda-device 0
                --output-file ${MODEL_DIR}/output_test.jsonl \
                --use-dataset-reader \
                ${MODEL_DIR}/model.tar.gz \
                "sharedtask2019/data/${CORPUS}/${CORPUS}_test.conll"
        python scripts/score_binary_output ${MODEL_DIR}/output_test.jsonl > ${MODEL_DIR}/metrics_test.txt
done
