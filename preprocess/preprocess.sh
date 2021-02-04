#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=/nfs/users/xueyou/data/corpus/dureader_v2/preprocessed
export OUT_DIR=/nfs/users/xueyou/data/corpus/dureader_v2/extracted

paragraph_extraction ()
{
    SOURCE_DIR=$1
    TARGET_DIR=$2
    echo "Start paragraph extraction, this may take a few hours"
    echo "Source dir: $SOURCE_DIR"
    echo "Target dir: $TARGET_DIR"
    mkdir -p $TARGET_DIR/trainset
    mkdir -p $TARGET_DIR/devset
    # mkdir -p $TARGET_DIR/testset

    echo "Processing trainset"
    cat $SOURCE_DIR/trainset/search.train.json | python paragraph_extraction.py train \
            > $TARGET_DIR/trainset/search.train.json

    echo "Processing devset"
    cat $SOURCE_DIR/devset/search.dev.json | python paragraph_extraction.py dev \
            > $TARGET_DIR/devset/search.dev.json

    # echo "Processing testset"
    # cat $SOURCE_DIR/testset/test.json | python paragraph_extraction.py test \
    #         > $TARGET_DIR/testset/search.test.json
    echo "Paragraph extraction done!"
}


PROCESS_NAME="$1"
case $PROCESS_NAME in
    --para_extraction)
    # Start paragraph extraction 
    if [ ! -d ${DATA_DIR} ]; then
        echo "Please download the preprocessed data first (See README - Preprocess)"
        exit 1
    fi
    paragraph_extraction ${DATA_DIR} ${OUT_DIR}
    ;;
    *)
        echo $"Usage: $0 {--para_extraction}"
esac
