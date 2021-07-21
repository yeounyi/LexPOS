## Generating Slogans with Linguistic Constraints
To implement the linguistic features of slogans such as repetition, here's a sequence-to-sequence transformer model <b>generating slogans with lexical and POS constraints</b>. You can specify certain keywords to be included and the desired syntax by list of POS tags. LexPOS finds words phonetically and semantically similar with user keywords and generates slogans including these words. The generated slogans also follow the given syntax. Furthermore, you can apply additional <b>phonetic constraints </b> implemented by adjusted logits distribution. 


### 1. Generating slogans with lexical, POS constraints 
#### 1.1. Code 
```python3
# clone this repo
git clone https://github.com/yeounyi/LexPOS
cd LexPOS
# generate slogans 
python3 generate_slogans.py -keywords cake -num_beams 3 -temperature 1.2
```
- `-keywords`: Keywords that you want to be included in slogans. You can enter multiple keywords, delimited by comma
-  `-pos_inputs`: You can either specify the particular list of POS tags delimited by comma, or the model will generate slogans with the most frequent syntax used in corpus. POS tags should follow the format of [Universal POS tags](https://universaldependencies.org/u/pos/).  
- `-num_beams`: Number of beams for beam search. Default to 1, meaning no beam search.
- `-temperature`: The value used to module the next token probabilities. Default to 1.0.
- `-model_path`: Path to the pretrained model

#### 1.2. Examples 

<b>Keyword</b>: `cake` <br>
<b>POS</b>: `[VERB, DET, NOUN, PUNCT, VERB, DET, NOUN, PUNCT]`	 <br>
<b>Output</b>: Bake a cake, bake a smile <br>

<b>Keyword</b>: `computer` <br>
<b>POS</b>: `[ADJ, NOUN, PUNCT, ADJ, NOUN, PUNCT]` <br>
<b>Output</b>: Superior Computer. Super peculiar. <br>

<b>Keywords</b>: `comfortable, furniture` <br>
<b>POS</b>: `[NOUN, ADP, NOUN, PUNCT, NOUN, ADP, NOUN, PUNCT]` <br>
<b>Output</b>: Signature of comfortable furniture. Signature of style. <br>


### 2. Generating slogans with lexical, POS, and phonetic constraints
#### 2.1. Code 
```python3
# clone this repo
git clone https://github.com/yeounyi/LexPOS
cd LexPOS/Phonetic_Constraints_During_Inference
# generate slogans 
python3 generate_slogans_with_phonetic_constraints.py -keywords vacation,island -num_beams 3 -temperature 1.2 -alpha 10
```
- `-keywords`: Keywords that you want to be included in slogans. You can enter multiple keywords, delimited by comma
-  `-pos_inputs`: You can either specify the particular list of POS tags delimited by comma, or the model will generate slogans with the most frequent syntax used in corpus. POS tags should follow the format of [Universal POS tags](https://universaldependencies.org/u/pos/).  
- `-num_beams`: Number of beams for beam search. Default to 1, meaning no beam search.
- `-temperature`: The value used to module the next token probabilities. Default to 1.0.
- `-model_path`: Path to the pretrained model
- `-alpha`: Weights of phonetical constraints. Default to 0.0, meaning no phonetic constraints.

#### 2.2. Examples

<b>Keywords</b>: `vacation,island` <br>
<b>POS</b>: `[VERB, DET, NOUN, PUNCT, VERB, DET, NOUN, PUNCT]`	 <br>
<b>alpha 0</b>: Make an occasion, vacation the island. <br>
<b>alpha 10</b>: Take an island vacation, every occasion. <br>

<b>Keywords</b>: `water,pollution` <br>
<b>POS</b>: `[NOUN, ADP, NOUN, PUNCT, NOUN, ADP, NOUN, PUNCT]`	 <br>
<b>alpha 0</b>: Pollution, without water, without depletion. <br>
<b>alpha 10</b>: Pollution of water, depletion of water. <br>

### Model Architecture
<br>
<img src="https://github.com/yeounyi/LexPOS/blob/main/assets/adj_graph.png" width=400>

### Pretrained Models 
1. `LexPOSBart`
- Trained with 2 lexical constraints 
- Trained with lexical constraints with fixed order 
2. `LexPOSBart_multi`
- Trained with 2-4 lexical constraints
- Trained with shuffled lexical constraints   

### References
https://github.com/aparrish/phonetic-similarity-vectors/

### Citation
```
@misc{yi2021lexpos,
  author = {Yi, Yeoun},
  title = {Generating Slogans with Linguistic Constraints using Sequence-to-Sequence Transformer},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yeounyi/LexPOS}}
}
```
