import argparse
import torch
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from LexPOSBartForConditionalGeneration import LexPOSBartForConditionalGeneration
from similar_words.get_similar_words import get_phon_and_sem_similar_words

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
tokenizer.add_tokens('<name>')

pos_tokenizer = BartTokenizer(vocab_file='POSvocab.json', merges_file='merges.txt')

def generate_slogans(input_words, model_path, pos_inputs=None, num_beams=1, temperature=1.0):
    # load model
    model = LexPOSBartForConditionalGeneration.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # search for phonetically & semantically similar words to each input word
    sim_words = []
    for word in input_words:
        sim_words += get_phon_and_sem_similar_words(word)[:3]

    # in case no particular POS constraints are given
    if not pos_inputs:
        lexical_inputs = []
        for word in sim_words:
            lexical_inputs += ['<mask> ' +  ' <mask> '.join(input_words + [word]) + ' <mask>'] * 3
        # popular list of POS tags in slogan data
        # [['VERB', 'DET', 'NOUN', 'PUNCT', 'VERB', 'DET', 'NOUN', 'PUNCT']]
        # [['NOUN', 'ADP', 'NOUN', 'PUNCT', 'NOUN', 'ADP', 'NOUN', 'PUNCT']]
        # [['DET', 'NOUN', 'ADP', 'NOUN', 'PUNCT', 'DET', 'NOUN', 'ADP', 'NOUN', 'PUNCT']]
        # [['ADJ', 'NOUN', 'PUNCT', 'ADJ', 'NOUN', 'PUNCT']]
        pos_inputs = [['VERB', 'DET', 'NOUN', 'PUNCT', 'VERB', 'DET', 'NOUN', 'PUNCT'], ['NOUN', 'ADP', 'NOUN', 'PUNCT', 'NOUN', 'ADP', 'NOUN', 'PUNCT'], ['ADJ', 'NOUN', 'PUNCT', 'ADJ', 'NOUN', 'PUNCT']] * (len(lexical_inputs) // 3)

    # in case particular POS constraints are given
    else:
        lexical_inputs = []
        for word in sim_words:
            lexical_inputs += ['<mask> ' +  ' <mask> '.join(input_words + [word]) + ' <mask>']
        pos_inputs = [pos_inputs] * len(lexical_inputs)

    # generate slogans
    preds = []
    for lexical_input, pos_input in zip(lexical_inputs, pos_inputs):
        inputs = tokenizer(lexical_input, return_tensors="pt").to(device)
        pos_inputs = pos_tokenizer.encode_plus(pos_input, is_pretokenized=True, return_tensors='pt').to(device)
        outputs = model.generate(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask,
                                 pos_input_ids = pos_inputs.input_ids, \
                                 pos_attention_mask = pos_inputs.attention_mask, num_beams=num_beams, temperature=temperature)

        for i in range(len(outputs)):
            preds.append(tokenizer.decode(outputs[i], skip_special_tokens=True))

    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-keywords', dest='keywords', help='input keywords to generate slogan with, delimited by comma')
    parser.add_argument('-pos_inputs', help='list of POS tags delimited by comma', type=str)
    parser.add_argument('-num_beams', help='Number of beams for beam search. Default to 1.', type=int, default=1)
    parser.add_argument('-temperature', help=' The value used to module the next token probabilities. Default to 1.0.', type=float, default=1.0)
    parser.add_argument('-model_path', type=str, default="./LexPOSBart_multi")
    args = parser.parse_args()

    if args.keywords:
        pos_inputs = None
        if args.pos_inputs:
            pos_inputs = [str(tag) for tag in args.pos_inputs.split(',')]
        preds = generate_slogans(input_words=str(args.keywords).split(','), model_path=args.model_path, pos_inputs=pos_inputs, num_beams=args.num_beams, temperature=args.temperature)
        print(preds)

# CUDA_VISIBLE_DEVICES=0 python3 generate_slogans.py -keywords cake -pos_inputs "VERB,DET,NOUN,PUNCT,VERB,DET,NOUN,PUNCT"
