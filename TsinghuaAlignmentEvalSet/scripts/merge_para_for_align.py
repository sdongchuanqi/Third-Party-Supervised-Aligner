import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--src_file", type=str)
parser.add_argument("--tgt_file", type=str)
parser.add_argument("--out_file", type=str)
parser.add_argument("--max_len", type=int, default=256)
parser.add_argument("--min_len", type=int, default=0)
parser.add_argument("--truc", action="store_true")
args = parser.parse_args()

# 将提取出来的伪平行文件通过 '|||' 合并，用于对齐

with open(args.src_file) as f:
    s_sents = [l.rstrip() for l in f]
try:
    with open(args.tgt_file) as f:
        t_sents = [l.rstrip() for l in f]
except:
    with open(args.tgt_file, encoding='ISO-8859-1') as f:
        t_sents = [l.rstrip() for l in f]    

with open(args.out_file, mode='w') as f:
    for i in range(len(s_sents)):
        if args.truc:
            if args.min_len <= len(s_sents[i].split()) <= args.max_len and args.min_len <= len(t_sents[i].split()) <= args.max_len:
                f.write(s_sents[i] + ' ||| ' + t_sents[i] + '\n')
        else:
            f.write(s_sents[i] + ' ||| ' + t_sents[i] + '\n')


