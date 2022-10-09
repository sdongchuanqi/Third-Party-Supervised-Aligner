import sys
from tqdm import tqdm
from transformers import AutoTokenizer


def main(bert_model, input_align, input_file1, input_file2, output_align):

    print(bert_model)
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
        token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
        token_src_len, token_tgt_len = [len(_) for _ in token_src], [len(_) for _ in token_tgt]
        res=[]
        for align in aligns:
            align = align.split('-')
            x,y = int(align[0])-1,int(align[1])-1
            presumx = 0 if x==0 else sum(token_src_len[:x-1])
            presumy = 0 if y==0 else sum(token_tgt_len[:y-1])
            for x_bpe in range(presumx[x]):
                for y_bpe in range(presumy[y]):
                    res.append(str(presumx+1)+'-'+str(presumy+1))
        res = ' '.join(res)
        output_align.write(res+'\n')

    file1.close()
    file2.close()
    align_file.close()
    output_align.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],sys.argv[4], sys.argv[5])