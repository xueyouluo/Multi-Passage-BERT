#!/usr/bin/env bash
export BERT_DIR=/data/xueyou/data/bert_pretrain/chinese_L-12_H-768_A-12
export RAW_DATA_DIR=/nfs/users/xueyou/data/corpus/dureader_v2/multi_passage
export OUTPUT_DIR=/nfs/users/xueyou/data/corpus/dureader_v2/multi_passage/mp_bert
export DATA_DIR=/nfs/users/xueyou/data/corpus/dureader_v2/multi_passage/data

python run_multi_passage_bert.py \
	--vocab_file=${BERT_DIR}/vocab.txt \
	--bert_config_file=${BERT_DIR}/bert_config.json \
	--init_checkpoint=${BERT_DIR}/bert_model.ckpt \
	--do_predict=True \
	--predict_file=${RAW_DATA_DIR}/dev_v2.json \
	--predict_dir=${OUTPUT_DIR} \
	--max_answer_length=200 \
	--predict_batch_size=2 \
	--max_seq_length=512 \
	--doc_stride=128 \
	--output_dir=${OUTPUT_DIR} \
	--data_dir=${DATA_DIR} \
	--do_lower_case=True \