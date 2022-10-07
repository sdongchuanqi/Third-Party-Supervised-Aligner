# Third-Party-Supervised-Wordaligner
This is the implementation of our work Third-Party Supervised Fine-tuning for Neural Word Alignments.

### Introduction
This work offer an simple and effective way to boost the existed aligner.With the signals from others aligners, pretrained model achieved lower AER.


## Usage
### Data Preparation
To get the data used in our paper, you can follow the instructions in [https://github.com/lilt/alignment-scripts](https://github.com/lilt/alignment-scripts).

### Get Subword-level Supervised Alignment From Other Aligner

In our preliminary experiment,to better use the alignment from the third party aligner,you have to get the alignment in the subword level.You need to tokenizer the words into subwords which is used by the pretrained model to be finetuned.

Here we offer an simply version that tokenizer the word into subword. 

Then follow the guidence for the third party aligner. 

Note that, some aligners usually convert subword alignment results to word alignment results,but you shouldn't convert subword alignment to word alignment.  

Here we offer an subword alignment result coming from Maskalign[https://github.com/THUNLP-MT/Mask-Align] which is used to finetune the mbert(https://huggingface.co/bert-base-multilingual-cased) 

### Finetune the pretrained model


 
