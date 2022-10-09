
for type in bert;do
num=6,7
layers=8
epochs=10
output_dir=checkpoint               #checkpoint
mkdir -p output_dir
pretrain_model=the_place_of_downloaded_pretrainedmodel_of_mbert # https://huggingface.co/bert-base-multilingual-cased
cache_path=cacheplace_for_next_load
 
CUDA_VISIBLE_DEVICES=$num  torchrun --nnodes=1 --nproc_per_node=2 \
     ../src/align_train.py\
    --train_file ../8w/chen.sents\
    --train_bpe_gold_file ../8w/8w.word.align.mbert.zeros\
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
