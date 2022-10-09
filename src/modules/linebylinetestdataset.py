import os
import torch
from tqdm import tqdm
import itertools
import random


from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# construst dataset for word align

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path, gold_path, gold_bpe_path, logger, max_length=128, min_length=1):
        assert os.path.isfile(file_path)
        logger.info("Creating features from dataset file at %s", file_path)
        cache_fn = args.cache_data
        if args.cache_data and os.path.isfile(cache_fn) and not args.overwrite_cache:
            logger.info("Loading cached data from %s", cache_fn)
            self.examples = torch.load(cache_fn)
        else:
            # Loading text data
            self.examples = []
            with open(file_path) as f:
                lines = f.readlines()
            
            # Loading bpe level gold data
            if gold_bpe_path is not None:
                assert os.path.isfile(gold_bpe_path)
                logger.info("Loading gold bpe level alignments at %s", gold_path)
                with open(gold_bpe_path) as f:
                    gold_bpe_lines = f.readlines()
                    
                # assert len(gold_bpe_lines) == len(lines)

            # Loading gold data
            if gold_path is not None:
                assert os.path.isfile(gold_path)
                logger.info("Loading gold alignments at %s", gold_path)
                with open(gold_path, encoding="utf-8") as f:
                    gold_lines = f.readlines()
                    
                # assert len(gold_lines) == len(lines)
            gold_bpe_id = 0
            gold_id = 0
            for line_id, line in tqdm(enumerate(lines), desc='Loading data', total=len(lines)):
                if len(line) > 0 and not line.isspace() and len(line.split(' ||| ')) == 2:
                    try:
                        src, tgt = line.split(' ||| ')
                        if src.rstrip() == '' or tgt.rstrip() == '':
                            logger.info("Skipping instance %s", line)
                            continue
                    except:
                       
                        logger.info("Skipping instance %s", line)
                        continue
                    sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
                    if len(sent_src) < min_length or len(sent_tgt) < min_length:
                        logger.info("Skipping instance %s", line)
                        continue
                    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
                    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
                    
                    tokenizer.src_lang = 'en_XX'
                    ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', max_length=max_length, prepend_batch_axis=True)['input_ids']            
                    tokenizer.src_lang = 'de_DE'
                    ids_src = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', max_length=max_length, prepend_batch_axis=True)['input_ids']
                    if len(ids_src[0]) == 1 or len(ids_tgt[0]) == 1:
                 
                        logger.info("Skipping instance %s", line)
                        continue

                    bpe2word_map_src = []
                    for i, word_list in enumerate(token_src):
                        bpe2word_map_src += [i for x in word_list]
                    bpe2word_map_tgt = []
                    for i, word_list in enumerate(token_tgt):
                        bpe2word_map_tgt += [i for x in word_list]

                    if gold_bpe_path is not None:
                        # word alignments at bpe level
                        try:
                            gold_bpe_line = gold_bpe_lines[gold_bpe_id].strip().split()
                            gold_bpe_pairs = []
                            for src_tgt in gold_bpe_line:
                                if 'p' in src_tgt:
                                    if args.ignore_possible_alignments:
                                        continue
                                    wsrc, wtgt = src_tgt.split('p')
                                else:
                                    wsrc, wtgt = src_tgt.split('-')
                                wsrc, wtgt = (int(wsrc), int(wtgt)) if not args.model_type == 'bert' else (int(wsrc)+1, int(wtgt)+1)
                                gold_bpe_pairs.append( (wsrc, wtgt) )
                            gold_bpe_id += 1
                        except:
                            logger.info("Error when processing the gold bpe level alignment %s, skipping", gold_bpe_lines[gold_bpe_id].strip())
                            gold_bpe_id += 1
                            continue
                        
                    else:
                        gold_bpe_pairs = None

                    if gold_path is not None:
                        # word alignments at bpe level
                        try:
                            gold_line = gold_lines[gold_id].strip().split()
                            gold_word_pairs = []
                            for src_tgt in gold_line:
                                if 'p' in src_tgt:
                                    if args.ignore_possible_alignments:
                                        continue
                                    wsrc, wtgt = src_tgt.split('p')
                                else:
                                    wsrc, wtgt = src_tgt.split('-')
                                wsrc, wtgt = (int(wsrc), int(wtgt)) if not args.gold_one_index else (int(wsrc)-1, int(wtgt)-1)
                                gold_word_pairs.append( (wsrc, wtgt) )
                            self.examples.append( (ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, gold_word_pairs, gold_bpe_pairs) )
                        
                            gold_id += 1
                        except:
                            logger.info("Error when processing the gold alignment %s, skipping", gold_lines[gold_id].strip())
                            gold_id += 1
                            continue
                    else:
                        self.examples.append( (ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, None, gold_bpe_pairs) )

            if args.cache_data:
                logger.info("Saving cached data to %s", cache_fn)
                torch.save(self.examples, cache_fn)

    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        return list(self.examples[index])
    '''
    #
    def __getitem__(self, i):
        neg_i = random.randint(0, len(self.examples)-1)
        while neg_i == i:
            neg_i = random.randint(0, len(self.examples)-1)
        return tuple(list(self.examples[i]) + list(self.examples[neg_i][:2]) )
    '''