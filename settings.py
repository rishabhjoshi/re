
class Setting(object):
    def __init__(self):
        self.voc_size = 114042
        self.num_steps = 70         # the number of unrolled steps of LSTM/GRU
        self.num_epochs = 40         #change it to 10, later 
        self.num_classes = 56       #output label
        self.gru_size = 350         #lstm: the hidden cell number
        self.keep_prob = 0.5        #the probability of keeping weights in dropout layer
        self.num_layer = 1 
        self.pos_size = 5           #
        self.pos_num = 123          #
        self.batch_size = 50           #change it to "batch_size". Entity pairs of each batch (batch size)
        #lr_decay = 0.5             #learning rate decay
        self.learning_rate = 0.0010  #empirically, 0.0001 is better
