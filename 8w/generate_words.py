import sys
from tqdm import tqdm
from transformers import AutoTokenizer

#import pdb
def main(input_file1, input_file2, input_align,out_words):
    #pdb.set_trace()
    bert_model='/data4/jpzhang/lexicon_induction/pretrain_model/bert-base-multilingual-cased'
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
    file1 = open(input_file1,'r')
    file2 = open(input_file2,'r')
    align_file = open(input_align,'r')
    out_words = open(out_words,'w')

    for line in tqdm(file1):
        src = line
        tgt = file2.readline()
        aligns = align_file.readline().strip().split()
        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        for align in aligns:
            if 'p' in align:
                continue
            align = align.split('-')
            x,y = int(align[0])-1,int(align[1])-1
            out_words.write(sent_src[x]+" ||| "+sent_tgt[y]+'\n')
        # print(output_align)
        # break
    file1.close()
    file2.close()
    align_file.close()
    out_words.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],sys.argv[4])