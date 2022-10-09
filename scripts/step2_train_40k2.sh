# 利用提取的对齐进行训练
# python -m torch.distributed.launch --nproc_per_node=2
# 输入的文本文件是未bpe， 对齐文件时bpe后的

for type in bert;do
num=6,7
layers=8
epochs=10
output_dir=/data4/cqdong/my_align/checkpoints_zh_en_dot_bert_1e-5_gold
mkdir -p output_dir
#pretrain_model=/data4/jpzhang/lexicon_induction/pretrain_model/xlm-mlm-xnli15-1024
#pretrain_model=/data4/cqdong/my_align/xlm-15
pretrain_model=/data4/jpzhang/lexicon_induction/pretrain_model/bert-base-multilingual-cased
cache_path=/data4/cqdong/my_align/cache/8w.zhen_gold_${type}\.pt
# torchrun --nproc_per_node=2     ----simalign \       --learning_rate 1e-5\  --overwrite_cache \ --rdzv_backend=c10d --rdzv_endpoint='127.0.0.1:7891' 
# CUDA_VISIBLE_DEVICES=$num python \
CUDA_VISIBLE_DEVICES=$num  torchrun --nnodes=1 --nproc_per_node=2  --rdzv_endpoint='127.0.0.1:7892'\
     /data4/cqdong/my_align/src/align_train.py\
    --train_file /data4/cqdong/my_align/zh2en/8w/chen.sents\
    --train_bpe_gold_file /data4/cqdong/my_align/zh2en/8w/8w.word.align.mbert.zeros\
    --model_name_or_path $pretrain_model\
    --tokenizer_name $pretrain_model\
    --config_name $pretrain_model\
    --learning_rate 1e-5\
    --overwrite_cache \
    --output_dir $output_dir\
    --block_size 128\
    --per_device_train_batch_size 16\
    --cache_data $cache_path \
    --num_train_epochs $epochs\
    --encoder_layers $layers \
    --model_type bert \
    --do_train 
done