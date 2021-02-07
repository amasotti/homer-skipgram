# SkipGram model for Ancient Greek

My implementation (in Pytorch `1.7.0`) of the Word2Vec model (Skip Gram Negative) trained on the homeric texts.

The main goal was for me to improve my programming skills, have fun with ancient greek texts and try to understand better how such models work. The project is by no mean intended to be complete or bug-free. This being said, feel absolutely free to copy, share, make suggestions or contact me. I would appreciate your feedback!

## Project structure

- `data/` : raw data and backups of models, vocabs and lists
  - `assets/` : summaries and plots
  - `models/` : model backup, for further trainig + `Skipgram_Pytorch_0502_beta` : this model is the first one I started to train with very simple layers (just embeddings + 1 log_sigmoid). It is efficient, but I wanted to try more complex settings. + `Skipgram_Pytorch_0502_gamma` : the second model I've trained. This model has a Dropout layer after the initialization of the input embds, that should improve the generalizability of a model.
  - `raw_data/` : the text files with the original homeric texts, extracted from Perseus Library
  - `vocabs/` : vocabs and json backups
- `preprocessing/` : python scripts for the initial stages of the NLP pipeline (tokenization, stopword deletion etc...)
- `utils/` : auxiliary files and script, mostly used to build the ngrams and the Dataset for the NN
- `skipgram_homer.py` : the main file, containing the script to train the neural model and some auxiliary functions (dataset constructor)
- `train_skipgram.py` : the main access point of the project. It contains the customizable parameters and the training loop.
- `skipgram_eval.ipynb` : some explorations and evaluations of the model (analogy, vector space, similarity queries...)

## How to run

The model was already tested and trained but is still a kind of work in progress. At the moment it lacks a user interface. But you can still run the python scripts in your terminal / IDE:

**Build the corpus**:

Make sure to have a cleaned raw_text (see [my other repo](https://github.com/amasotti/AncientGreek_NLP) for some tools). Then you can tokenize the text and build the vocabularies by running:

      python3 ./preprocessing/preprocessing.py

just edit the `args Namespace` with your own paths:

```python
args = Namespace(
    raw_data="../data/raw_data/HomerGesamt_deaccented.txt",
    word2index="../data/vocabs/Homer_word2index.json",
    word_frequencies="../data/vocabs/Homer_word_frequencies.json"
)
```

`raw_data` points to the raw text, the other two are paths to save the vocabularies created.

**Build the SkipGram dataset**

The function `skip_gram_dataset(corpus, word2index, window=5)` in `utils/dataset.py` builds the skipgram trainable dataset. It requires a corpus with tokenized sentences (see above), the word2index lookup dictionary and the window size (an integer of your choice).

** Run the model **

`skipgram_homer.py` contains all the necessary functions to train the neural network.

`train_skipgram.py` : You can adjust the hyperparameters in the argparse and paths Namespaces a the beginning of the file and run this file to train the model.
The model will be saved automatically if the loss improves. At the end of the traing phase it will show a plot of the loss/batches.
While training it will display (once in 1500 minibatches, customizable) the most similar words for a small set of test words declared in the main file. I found this a nice feature to implement, since it allows you to see improvements (ideally) in real-time.

** Explore the results **

Have a look at the Jupyter Notebook `skipgram_eval.ipynb`.

## Important hyperparameters

The model implemented here is quite a vanilla Word2Vec Skipgram. I've however tried to improve some parameters taking into account some suggestions from the literature:

- Subsampling : instead of using pure counts/frequency based vocabularies for the model, I've implemented the sub-sampling formula according to which the P value for each token is the following:

                P(w) = 1+ sqrt(freq(w) / 10e-3)) * (10e-3 / freq(w))

  I've not used _subsampling_ as strategy to reduce the trainable data, since this would also penalize rare words, which could be very interesting in the case of Homer. I've used _subsampling_ to construct the noise distribution used to derive the negative samples.

- Split sets (training, validation): common practice in almost all Neural Networks implementations, but still I've seen many projects with just one set (training) implemented. The main purpose was to avoid overfitting.

- Negatives: in each steps the models changes the values for a small (15) number of negative context words (see paper by Levy & Goldberg).

- Scheduler : the learning rate is not constant. The `ReduceLROnPlateau` scheduler (factor=`0.3`, patience=`1`) changes the learning rate if the loss doesn't improve.

- Dropout layer added : some studies suggest that a further `F.dropout()` layer after the initialization of the input embeddings can help to avoid overfitting. The dropout layer just masks some random values in the tensor with a probability p passed as parameter.

#### Training phase (Loss)

<img src="./data/assets/losses_train_RMSop_optimizer.png" alt="loss_train" style="width: 500px; height:250px" >

## Predictions

Probable context words for the header word:

| γλαυκῶπις |  εὔχομαι  |     ἔρος      |  ερχομαι  |
| :-------: | :-------: | :-----------: | :-------: |
|    θεά    |   ειναι   | περιπροχυθεις |  δομενευ  |
|   ἀθήνη   |  εξειπω   |    ρυμνης     |  γενεσιν  |
|  Παλλὰς   | νικησαντʼ |   γυναικος    | οτρυνῃσιν |
|  ἐνόησε   |    εγω    |  καταλεξομεν  |  απασης   |
| ἠμείβετʼ  |    ος     |     ουδε      | βουληφορε |

## Requirements

see the `requirements.txt` (**NB:** the file was automatically generated with `pip freeze` and contains many dependencies, that are not strictly needed).

```

```
