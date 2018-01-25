

# RE
Relation extraction has been widely used to extract new relational facts from open corpus. Previous relation extraction methods are faced with the problem of wrong labels and noisy data, which substantially decrease the performance of the classifier. In this paper, we propose an ensemble neural networks model - Adaptive Boosting LSTMs with Attention, to more effectively perform relation extraction. Specifically, our model first employs the recursive neural network LSTMs to embed the latent semantics of each sentence. Then we import attention into LSTMs by considering that the words in a sentence do not contribute equally to the sentence representation. Next via adaptive boosting method, we build strategically several neural classifiers. The error feedback of the samples can be attained by the forward of neural networks. Adaptive boosting sets a constraint gradient descent of the neural networks in the loss function based on the error feedback. In this way, we can train a more robust neural network classifier. By ensembling multiple such LSTM classifiers with adaptive boosting, we could build a more effective joint ensemble neural network relation extractor. Experiment results on real dataset demonstrate the superior performance of the pro- posed model, improving F1-score by about 8% compared to the state-of-the-art models.

#Evaluation Results
P@N comparison:

![comparison](https://github.com/RE-2018/re/blob/master/result.png)



# Data
We use the same dataset(NYT10) as in [Lin et al.,2016]. And we provide it in origin_data/ directory. NYT10 is originally released by the paper "Sebastian Riedel, Limin Yao, and Andrew McCallum. Modeling relations and their mentions without labeled text."  

Pre-Trained Word Vectors are learned from New York Times Annotated Corpus (LDC Data LDC2008T19), which should be obtained from LDC (https://catalog.ldc.upenn.edu/LDC2008T19). And we provide it also in the origin_data/ directory.

To run our code, the dataset should be put in the folder origin_data/ using the following format, containing four files
- train.txt: training file, format (fb_mid_e1, fb_mid_e2, e1_name, e2_name, relation, sentence).
- test.txt: test file, same format as train.txt.
- relation2id.txt: all relations and corresponding ids, one per line.
- vec.txt: the pre-train word embedding file

Before you train your model, you need to type the following command:  
`python init.py`
to transform the original data into .npy files for the input of the network. The .npy files will be saved in data/ directory.


# Requirements
- Python (>=2.7)
- TensorFlow (>=1.0)
- scikit-learn (>=0.18)
- Matplotlib (>=2.0.0)

# Train
For training, you need to type the following command:  
`python train.py`  
The training model file will be saved in folder model/

You can lauch the tensorboard to see the softmax_loss, l2_loss and final_loss curve by typing the following command:
`tensorboard --logdir=./train_loss`  

# Test
For testing, you need to run the `test.py` to get all results on test dataset. BUT before you run it, you should change the pathname and modeliters you want to perform testing. 

As an example, we provide our best model in the model/ directory. You just need to type the following command:  
`python test.py`  

To draw the PR curve for the sample model, you just need to type the following command:  
`python plot_pr.py`  
The PR curve will be saved as .png in current directory. If you want to plot the PR curve for your own model, you just need to change the modeliters in the `plot_pr.py` 

