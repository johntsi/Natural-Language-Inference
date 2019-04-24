# Natural Language Inference and Transfer Tasks

### preprocess.py
*Provide the path for glove.840B.300d and the the snli_1.0 folder
(in the script, no argparser)

This will create the following in the ./../preprocessed_data folder
  1. embeddings_lower.csv (dataframe indexed with the lowercase tokens)
  2. vocabulary_lower.txt (all the lowercase tokens from the snli corpus)
  3. snli folder with the following dataframes (with columns [sentence1(tokenized), sentence2(tokenized), label, ntokens1, ntokens2])
    
    a. train_lower.csv 
    b. dev_lower.csv
    c. test_lower.csv
    
    
### map_unk_token.py
Runs through the tokenized sentencese in train, dev and test and maps the unknown token <unk> to all the tokens not found in the glove data. We can do really efficient batching by directly retrieving all the word embeddings for a whole batch of sentences in train.py
(4x faster than looping)
This will create the following in the ./../preprocessed_data/snli folder
   
    a. train_lower_mapped.csv 
    b. dev_lower_mapped.csv
    c. test_lower_mapped.csv
  
  
### data.py
Dataset object that takes care of loading the data (embeddings_lower.csv, vocabulary_lower.txt and mapped snli)
Contrains also get_batch function used in training
Contrains also two functions used for experimentation on different levels on ambiguity (nothing to do with the training process)


### models.py
Contains 4 class of models
  1. MLP classifier
  2. Bag-of-Words (BoW) encoder
  3. LSTM encoder (uni or bi)
  4. bi-LSTM encoder with max pooling
  
### train.py
Run: python.exe train.py --hyperparameter1=X --hyperparameter2=Y
Please refer to the argparser at the end of the script for a list
Also refer to the report for reproducing the results

Handles training procedure
#### Includes the following functions
  1. plot_grad_flow (gradient flow visualization)
  2. accuracy
  3. training_epoch (one pass of the whole dataset)
  4. evaluation_step (evaluation on either dev or test set)
  
#### Produces the following
  1. Best model (dev set accuracy)
  2. Best encoder (dev set accuracy) (encoder != BoW)
  3. Figure with learning curves
  4. Folder with figures of gradient flow during the training


### evaluate.py
Run: python.exe evaluate.py --task=... --encoder_name=... --path_to_vec=... --path_to_senteval=...
Please refer to the argparser
Erase the line where I force the path of the model based on the encoder_name if you are not using my own model
By default I am using the hyperparameters I used to produce the models, chnage them accordingly
 
 Loads checkpoint and uses encoder for the task specificied
 Most of the file is the same as found online in SentEval
 Changed only the batcher
 
 
 ### infer.py
Run: python.exe infer.py --encoder_name=...
Please refer to the argparser
Erase the line where I force the path of the model based on the encoder_name if you are not using my own model
By default I am using the hyperparameters I used to produce the models, chnage them accordingly
 
 Interactive session for demonstrating the NLI task with a pre-trained model
 
### eval_snli.py
Script for experimenting with different levels of ambiguity on the different models produced in this assignment
