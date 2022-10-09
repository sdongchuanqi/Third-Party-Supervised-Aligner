# 将对齐文件转换为标准格式的对齐文件输出
import argparse
import re
parser = argparse.ArgumentParser(description='change format')
parser.add_argument('--in_file', type=str, default=None, 
                    help='source file')
parser.add_argument('--out_file', type=str, default=None,
                    help='align file')
args = parser.parse_args()

in_f = open(args.in_file)
out_f = open(args.out_file, mode='w')
def return_symbol(x):
    return '-' if x == '1' else 'p'

for line in in_f:
    
    orign_line = [re.split(r'[:/]', pair)  for pair in line.strip().split(' ') if pair]
    try: 
        out_line = ' '.join([f'{tup[0]}{return_symbol(tup[2])}{tup[1]}' for tup in orign_line])
    except:
        print(orign_line,"sss"+line+"ssd")
        exit()
    out_f.write(out_line + '\n')

in_f.close()
out_f.close()