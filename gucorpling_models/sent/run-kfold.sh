#!/bin/bash

#org data dir
DATA_DIR="/disk1/shabnam/codes/data/dstrpt/2021"
INFO="/disk1/shabnam/codes/data/dstrpt/2021-info"
FOLDS_DIR="/disk1/shabnam/codes/data/dstrpt/2021-folds"
PRETRAINED_DIR="/disk1/shabnam/codes/data/dstrpt/pretrained/"

fold="0"

python get_docs_info.py -d $DATA_DIR/ -o $INFO/org/

python build_kfold_data.py -d $DATA_DIR/ -i $INFO/org/ -o $FINAL_DIR/

python build_sent_data.py -d $FOLDS_DIR/$fold/ -o $FOLDS_DIR/sentencer-$fold/

python flair_splitter.py $FOLDS_DIR/sentencer-$fold/ -m train | tee trainlogsf$fold.txt

python get_docs_info.py -d $FOLDS_DIR/sentencer-$fold/ -o $INFO/$fold/

python conll2sgml.py -d $FOLDS_DIR/sentencer-$fold/ -o $FOLDS_DIR/sentencer-$fold/

python process_predictions.py -f $FOLDS_DIR/sentencer-$fold/ -m test

python python postprocessing.py -f $FOLDS_DIR/sentencer-$fold/ -i $INFO/$fold/ -d $PRETRAINED_DIR -m test