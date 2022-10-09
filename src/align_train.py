
import argparse
import logging
import math
import tqdm
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from torch.nn.utils.rnn import pad_sequence
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
# jpzhang 2021.12.16
from modules import LineByLineTextDataset
from modeling_mbart_align import MBartEncoderAlign
from modeling_mbert_align import MBertForAlign
logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

#https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling/requirements.txt

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

local_rank = int(os.environ["LOCAL_RANK"])
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a align task")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--train_gold_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_gold_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--train_bpe_gold_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_bpe_gold_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=4000000,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=1000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--co_train",action="store_true",help="co_loss")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Do train?"
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Do eval?"
    )
    parser.add_argument(
        "--cache_data", type=str, default=None, help="load data cache from cache dir"
    )
    parser.add_argument(
        "--gold_one_index", action="store_true", help="Whether the gold alignment files are one-indexed"
    )
    parser.add_argument(
        "--ignore_possible_alignments", action="store_true", help="Whether the possible alignments are ignored"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--simalign", action="store_true", help="Use the simalign methods?")
    parser.add_argument("--encoder_layers", type=int, default=12)
    parser.add_argument("--self_train", action="store_true")
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def load_and_cache_examples(args, tokenizer, logger, evaluate=False):
    file_path = args.validation_file if evaluate else args.train_file
    gold_path = args.validation_gold_file if evaluate else args.train_gold_file
    gold_bpe_path = args.validation_bpe_gold_file if evaluate else args.train_bpe_gold_file
    return LineByLineTextDataset(tokenizer, args, file_path=file_path, gold_path=gold_path, gold_bpe_path=gold_bpe_path, logger=logger)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def main():

    args = parse_args()
    accelerator = Accelerator()
    #local_rank = args.local_rank
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    accelerator.wait_for_everyone()


    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        
        if args.model_type == 'mbart':
            config.encoder_layers = args.encoder_layers
            model = MBartEncoderAlign.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
        elif args.model_type == 'bert':
            config.num_hidden_layers = args.encoder_layers
            model = MBertForAlign.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                )
    else:
        logger.info("Training new model from scratch")
        if args.model_type == 'mbart':
            
            config.encoder_layers = args.encoder_layers
            model = MBartEncoderAlign.from_config(config)
        elif args.model_type == 'bert':
            model = MBertForAlign.from_config(config)
         
    # 为不同参数设置不同的decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (( not (any(nd in n for nd in no_decay)) ) and p.requires_grad)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if ( (any(nd in n for nd in no_decay)) and p.requires_grad)], "weight_decay": 0.0},
    ]
    def is_in_ml(n):
        return (n[0] < args.block_size) and (n[1] < args.block_size)
    def collate(examples):
        # return examples_src, examples_tgt, guides
        examples_src, examples_tgt = [], []
        src_len = tgt_len = 0
        bpe2word_map_src, bpe2word_map_tgt = [], []
        word_aligns = []
        word_bpe_aligns = []
        for idx, example in enumerate(examples):
            end_id = example[0][0][-1].view(-1)
            src_id = example[0][0][:args.block_size]
            src_id = torch.cat([src_id[:-1], end_id])
            tgt_id = example[1][0][:args.block_size]
            tgt_id = torch.cat([tgt_id[:-1], end_id])

            examples_src.append(src_id)
            examples_tgt.append(tgt_id)

            src_len = max(src_len, len(src_id))
            tgt_len = max(tgt_len, len(tgt_id))
    
            bpe2word_map_src.append(torch.Tensor(example[2]).type_as(src_id))
            bpe2word_map_tgt.append(torch.Tensor(example[3]).type_as(tgt_id))
            word_aligns.append(example[4]) # alignments at word level
            
            word_bpe_aligns.append(list(filter(is_in_ml, example[5]))) # alignments at bpe level

        examples_src = pad_sequence(examples_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        examples_tgt = pad_sequence(examples_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        bpe2word_map_src = pad_sequence(bpe2word_map_src, batch_first=True, padding_value=-1)
        bpe2word_map_tgt = pad_sequence(bpe2word_map_tgt, batch_first=True, padding_value=-1)
        # batch
    

        if word_aligns[0] is None:
            word_aligns = None
        else:
            word_aligns_lists = []
            for idx, aligns in enumerate(word_bpe_aligns):
                word_aligns_lists.extend([[idx, p[0], p[1]] for p in aligns])
            word_aligns = torch.Tensor(word_aligns_lists).type_as(examples_src)

        if word_bpe_aligns[0] is None:
            word_bpe_aligns = None
        else:
            word_bpe_aligns_lists = []
            max_src_len = examples_src.size(1)
            for idx, aligns in enumerate(word_bpe_aligns):
                word_bpe_aligns_lists.extend([[p[0] + idx * max_src_len, p[1]] for p in aligns])
            word_bpe_aligns = torch.Tensor(word_bpe_aligns_lists).type_as(examples_src)
        

        # return examples_src, examples_tgt, bpe2word_map_src, bpe2word_map_tgt, word_aligns, word_bpe_aligns
        if not args.self_train and not word_bpe_aligns is None:
            return examples_src, examples_tgt, bpe2word_map_src, bpe2word_map_tgt, word_bpe_aligns
        else:
            return examples_src, examples_tgt, bpe2word_map_src, bpe2word_map_tgt
        

    if args.do_train:
        #with accelerator.main_process_first():
        logger.info("loading data") 
        train_dataset = load_and_cache_examples(args, tokenizer, logger, evaluate=False)

        accelerator.wait_for_everyone()
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size, collate_fn=collate, num_workers=4
            )

        # construc optimizer
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

        t_total = len(train_dataloader) * args.num_train_epochs
        if args.max_step > 0 and args.max_step < t_total:
            t_total = args.max_step
            
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=t_total
        )
        logger.info("train data") 
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        train_epoch = args.num_train_epochs
        completed_step = 0  
        progress_bar = tqdm(range(len(train_dataloader) * train_epoch), disable=not accelerator.is_local_main_process)
        for iter_epoch in range(train_epoch):
            model.train()
            for step, batch in enumerate(train_dataloader):
                inputs_src, inputs_tgt = batch[0], batch[1]
                attention_mask_src, attention_mask_tgt = (inputs_src!=tokenizer.pad_token_id), (inputs_tgt!=tokenizer.pad_token_id)

                if args.self_train:
                    loss = model(inputs_src, attention_mask_src, inputs_tgt, attention_mask_tgt, word_bpe_align=None, simalign_method=args.simalign, self_train=True)
                else:
                    loss = model(inputs_src, attention_mask_src, inputs_tgt, attention_mask_tgt, word_bpe_align=batch[-1], simalign_method=args.simalign)
                if step % 100 == 0:
                    print(loss)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_step += 1
                if completed_step >= args.max_step:
                    break
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), args.output_dir+f'/model_{iter_epoch}.pt')

if __name__ == "__main__":
    main()