# python -m torch.distributed.launch --nproc_per_node=2
# 输入的文本文件是未bpe， 对齐文件时bpe后的
num=1
# if true; then
# validation_file=/data4/cqdong/Mask-Align/myjobs/data17/noprocess/jaen.test.sents
# validation_gold_file=/data4/cqdong/Mask-Align/myjobs/data17/noprocess/align-test.txt
# cache_data=/data4/cqdong/my_align/cache/tst_ja2en40k.pt
# output_dir=/data4/cqdong/my_align/output/ja2en40tst/
# fi
# # dev
# if false; then
# validation_file=/data4/cqdong/Mask-Align/myjobs/data17/noprocess/jaen.dev.sents
# validation_gold_file=/data4/cqdong/Mask-Align/myjobs/data17/noprocess/align-dev.txt
# cache_data=/data4/cqdong/my_align/cache/dev_ja2en40k.pt
# output_dir=/data4/cqdong/my_align/output/ja2en40dev/
# fi

# #dev
# if true; then
# validation_file=/data4/cqdong/my_align/zh2en/zh2en/TsinghuaAlignmentEvalSet/v1/devset/dev.ch2en
# validation_gold_file=/data4/cqdong/my_align/zh2en/zh2en/TsinghuaAlignmentEvalSet/v1/devset/dev.wa.formal.align
# cache_data=/data4/cqdong/my_align/cache/dev_ch2en40k.pt
# output_dir=/data4/cqdong/my_align/output/ch2en40dev/
# fi
# test
if true; then
validation_file=/data4/cqdong/my_align/zh2en/zh2en/TsinghuaAlignmentEvalSet/v1/tstset/tst.ch2en
validation_gold_file=/data4/cqdong/my_align/zh2en/zh2en/TsinghuaAlignmentEvalSet/v1/tstset/tst.wa.formal.align
cache_data=/data4/cqdong/my_align/cache/tst_de2en40k.pt
output_dir=/data4/cqdong/my_align/output/ch2en40tst/
fi

# if true; then
# validation_file=/data4/cqdong/my_align/deen/test/test.sents
# validation_gold_file=/data4/cqdong/my_align/deen/test/deen.talp
# cache_data=/data4/cqdong/my_align/cache/tst_de2en40k.pt
# output_dir=/data4/cqdong/my_align/output/de2en40tst/
# fi
# # dev
# if false; then
# validation_file=/data4/cqdong/my_align/deen/valid/valid.sents
# validation_gold_file=/data4/cqdong/my_align/deen/valid/maskalign_valid.txt
# cache_data=/data4/cqdong/my_align/cache/dev_de2en40k.pt
# output_dir=/data4/cqdong/my_align/output/de2en40dev/
# fi

# validation_file=/data4/cqdong/Mask-Align/myjobs/data12/noprocess/enfr.sents
# validation_gold_file=/data4/cqdong/Mask-Align/myjobs/data12/noprocess/enfr.talp
cache_data=/data4/cqdong/my_align/cache/tst_enfr40k.pt
output_dir=/data4/cqdong/my_align/output/enfr40tst/
mkdir -p output_dir
for check_num in {4..4};do
#    --simalign\
for layers in {8..8};do
CUDA_VISIBLE_DEVICES=$num python /data4/cqdong/my_align/src/align_extract.py \
    --validation_file  $validation_file \
    --validation_gold_file $validation_gold_file \
    --model_name_or_path /data4/cqdong/my_align/checkpoints_zh_en_dot_bert_1e-5_gold/model_${check_num}\.pt\
    --tokenizer_name /data4/jpzhang/lexicon_induction/pretrain_model/bert-base-multilingual-cased\
    --config_name /data4/jpzhang/lexicon_induction/pretrain_model/bert-base-multilingual-cased\
    --cache_data $cache_data \
    --overwrite_cache \
    --output_dir $output_dir \
    --block_size 512 \
    --encoder_layers $layers \
    --per_device_eval_batch_size 16 \
    --model_type bert \
    --do_eval 
#   --checkpoint_dir /data4/cqdong/my_align/checkpoints_zh_en/model_4.pt
    
# 测试对齐错误率
python /data4/cqdong/Mask-Align/alignment-scripts/scripts/aer.py\
    $validation_gold_file\
    $output_dir/align.txt\
    --oneRef --fAlpha 0.5 >>zh2en.out
    echo $layers>>zh2en.out
done
done