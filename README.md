## LexPOS
To implement the linguistic features of slogans such as repetition, here's a sequence-to-sequence transformer model <b>generating slogans with lexical and POS constraints</b>. You can specify certain keywords to be included and the desired syntax. LexPOS finds words phonetically and semantically similar with user keywords and generates slogans including these words. Furthermore, the generated slogans follow the given syntax.

### Usage 
```python3
# clone this repo
git clone https://github.com/yeounyi/LexPOS
cd LexPOS
# generate slogans 
python3 generate_slogans.py -keywords cake
```
- `-keywords`: Keywords that you want to be included in slogans. You can enter multiple keywords, delimited by comma
-  `-pos_inputs`: You can either specify the particular list of POS tags delimited by comma, or the model will generate slogans with the most frequent syntax used in corpus 
- `-num_beams`: Number of beams for beam search. Default to 1, meaning no beam search.
- `-temperature`: The value used to module the next token probabilities. Default to 1.0.
- `-model_path`: Path to the pretrained model

### Model Architecture
<br>
<img src="https://github.com/yeounyi/yeounyi.github.io/blob/main/assets/img/model_structure.JPG?raw=true" width=400>
<br>

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
  title = {LexPOS: Generating Slogans with Lexical and POS Constraints},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yeounyi/LexPOS}}
}
```
