# SkipGram model for Ancient Greek

My implementation (in Pytorch `1.7.0`) of the Word2Vec model (Skip Gram Negative) trained on the homeric texts.

The main goal was for me to improve my programming skills, have fun with ancient greek texts and try to understand better how such models work. The project is by no mean intended to be complete or bug-free. This being said, feel absolutely free to copy, share, make suggestions or contact me. I would appreciate your feedback!

## Project structure

- `data/` : raw data and backups of models, vocabs and lists
  - `assets/` : summaries and plots
  - `models/` : model backup, for further trainig
        + `Skipgram_Pytorch_0502_beta` : this model is the first one I started to train with very simple layers (just embeddings + 1 log_sigmoid). It is efficient, but I wanted to try more complex settings.
        + `Skipgram_Pytorch_0502_gamma` : the second model I've trained. This model has a Dropout layer after the initialization of the input embds, that should improve the generalizability of a model. 
  - `raw_data/` : the text files with the original homeric texts, extracted from Perseus Library
  - `vocabs/` : vocabs and json backups
- `preprocessing/` : python scripts for the initial stages of the NLP pipeline (tokenization, stopword deletion etc...)
- `utils/` : auxiliary files and script, mostly used to build the ngrams and the Dataset for the NN
- `skipgram_homer.py` : the main file, containing the script to train the neural model
- `skipgram_eval.ipynb` : some explorations and evaluations of the model (analogy, vector space, similarity queries...)

## Important hyperparameters

The model implemented here is quite a vanilla Word2Vec Skipgram. I've however tried to improve some parameters taking into account some suggestions from the literature:

- Subsampling : instead of using pure counts/frequency based vocabularies for the model, I've implemented the sub-sampling formula according to which the P value for each token is the following:

                P(w) = 1+ sqrt(freq(w) * 10e-3)) * 10e-3 / freq(w)

- Split sets (training, validation): common practice in almost all Neural Networks implementations, but still I've seen many projects with just one set (training) implemented.

- Negatives: in each steps the models changes the values for a small (15) number of negative context words (see paper by Levy & Goldberg).

- Scheduler : the learning rate is not constant. The `ReduceLROnPlateau` scheduler (factor=`0.3`, patience=`1`) changes the learning rate if the loss doesn't improve.

- Dropout layer added : some studies suggest that a further `F.dropout()` layer after the initialization of the input embeddings can help to avoid overfitting. The dropout layer
just masks some random values in the tensor with a probability p passed as parameter. 

#### Training phase (Loss)

<img src="./data/assets/losses_train_RMSop_optimizer" alt="loss_train" style="width: 500px; height:250px" >

### Predictions

|    θεά    |  εὔχομαι  |     ἔρος      |  ερχομαι  |
| :-------: | :-------: | :-----------: | :-------: |
| γλαυκῶπις |   ειναι   | περιπροχυθεις |  δομενευ  |
|   ἀθήνη   |  εξειπω   |    ρυμνης     |  γενεσιν  |
|    Ἥρη    | νικησαντʼ |   γυναικος    | οτρυνῃσιν |
|   θύμῳ    |    εγω    |  καταλεξομεν  |  απασης   |
|    περ    |    ος     |     ουδε      | βουληφορε |

## Requirements

see the `requirements.txt` (**NB:** the file was automatically generated with `pip freeze` and contains many dependencies, that are not strictly needed).
