import argparse
from tqdm import tqdm
from transformers import AutoTokenizer


def main(input_file1, input_file2, input_align,output_align,mode):
    bert_model='/data4/jpzhang/lexicon_induction/pretrain_model/bert-base-multilingual-cased'
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
    file1 = open(input_file1,'r')
    file2 = open(input_file2,'r')
    align_file = open(input_align,'r')
    output_align = open(output_align,'w')
    for line in tqdm(file1):
        src = line
        tgt = file2.readline()
        aligns = align_file.readline().strip().split()
        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()

        token_src, token_tgt = [bert_tokenizer.tokenize(word) for word in sent_src], [bert_tokenizer.tokenize(word) for word in sent_tgt]
        token_src_len, token_tgt_len = [len(_) for _ in token_src], [len(_) for _ in token_tgt]
        presum_src = [0 for _ in range(len(token_src_len))]
        for i in range(1,len(token_src_len)):
            presum_src[i]=presum_src[i-1]+token_src_len[i-1]

        presum_tgt = [0 for _ in range(len(token_tgt_len))]
        for i in range(1,len(token_tgt_len)):
            presum_tgt[i]=presum_tgt[i-1]+token_tgt_len[i-1]

        res=[]
        for align in aligns:
            if 'p' in align:
                continue
            align = align.split('-')
            x,y = int(align[0])-1,int(align[1])-1
            try :
                presumx = presum_src[x]
                presumy = presum_tgt[y]
            except:
                import pdb
                pdb.set_trace()
            for x_bpe in range(token_src_len[x]):
                for y_bpe in range(token_tgt_len[y]):
                    res.append(str(presumx+1+x_bpe)+'-'+str(presumy+1+y_bpe))


        res = ' '.join(res)
        output_align.write(res+'\n')
    file1.close()
    file2.close()
    align_file.close()
    output_align.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='word 2 wordpiece')
    parser.add_argument('--input_file1', type=str, default=None, help='source file')
    parser.add_argument('--input_file2', type=str, default=None, help='tgt file')
    parser.add_argument('--input_align', type=str, default=None, help='align word file')
    parser.add_argument('--output_align', type=str, default=None,help='align wordpiece file')
    parser.add_argument('--mode', type=str, default=None, help='mode full')
    args = parser.parse_args()

    main(args.input_file1,args.input_file2,args.input_align,args.output_align,args.mode)