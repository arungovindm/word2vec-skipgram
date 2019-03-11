# word2vec-skipgram
E1 246 Natural Language Understanding
assignment 1


For training the word2vec skipgram model run w2v.py with the following command line inputs
</br>
```bash
python w2.py batch_size negative_sample_count
```
To evaluate the model on SimLex-999 word similarity task run the following</br>

```bash
python simlex_eval.py path/to/word_vectors.dat path/to/SimLex-999.txt
