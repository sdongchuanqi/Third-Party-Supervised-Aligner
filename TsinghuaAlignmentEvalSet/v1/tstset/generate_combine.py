import sys
def generate_sentence(file1,file2,out):
    file1 = open(file1,'r')
    file2 = open(file2,'r')
    sentences1 = file1.readlines()
    sentences2 = file2.readlines()
    print(len(sentences1),len(sentences2))
    with open(out,'w') as f:
        for sent1,sent2 in zip(sentences1,sentences2):
            f.write(sent1.strip()+" ||| "+sent2.strip()+'\n')
    file1.close()
    file2.close() 
    
if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    out = sys.argv[3]
    generate_sentence(file1,file2,out)
