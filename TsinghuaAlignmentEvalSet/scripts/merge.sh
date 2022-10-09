out_sents=/data4/jpzhang/lexicon_induction/TsinghuaAlignmentEvalSet/v2/ch2en.sents
python /data4/jpzhang/lexicon_induction/TsinghuaAlignmentEvalSet/scripts/merge_para_for_align.py\
 --src_file /data4/jpzhang/lexicon_induction/TsinghuaAlignmentEvalSet/v2/chinese\
 --tgt_file /data4/jpzhang/lexicon_induction/TsinghuaAlignmentEvalSet/v2/english\
 --truc \
 --min_len 5\
 --max_len 128\
 --out_file $out_sents