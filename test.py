
import tensorflow as tf
import numpy as np
#from sklearn.metrics import average_precision_score
from sklearn import metrics

from sklearn.metrics import classification_report
from settings import Setting
from attention import Attention_layer
from model import GRU

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def test_func(test_y_file, word_file, pos1_file, pos2_file, ans_file, res_file):

    model_path = './model/ATT-MODEL'

    word_embedding = np.load('./data/vec.npy')
    wordpos1 = np.load('./data/wordpos1.npy')
    wordpos2 = np.load('./data/wordpos2.npy') 
    

    #test_y = np.load('./data/test_y.npy')
    #test_sentence_word = np.load('./data/test_sentence_word.npy')
    #test_sentence_pos1 = np.load('./data/test_sentence_pos1.npy')
    #test_sentence_pos2 = np.load('./data/test_sentence_pos2.npy')
    #all_ans = np.load('./data/all_ans.npy')

    test_y = np.load('./data/'+test_y_file)
    test_sentence_word = np.load('./data/'+ word_file)
    test_sentence_pos1 = np.load('./data/'+ pos1_file)
    test_sentence_pos2 = np.load('./data/'+ pos2_file)
    all_ans = np.load('./data/'+ans_file)

    all_alpha = np.load("./data/all_alpha.npy")
    
    
    settings = Setting()
    batch_size = settings.batch_size

    with tf.Graph().as_default():
       sess = tf.Session()

       with sess.as_default():
            print "model testing begin"
            with tf.variable_scope("model"):
                m = GRU(is_training=False, word_embeddings=word_embedding, word_pos1=wordpos1, word_pos2=wordpos2, settings=settings)
	    
            saver = tf.train.Saver()
	    
            i_length = int(len(test_sentence_word)/float(settings.batch_size))
            test_length = i_length * settings.batch_size

            ClassEst = np.zeros((test_length, settings.num_classes-1))

            
            for epoch in range(settings.num_epochs - 25, settings.num_epochs - 15):
                alpha = all_alpha[epoch]
                saver.restore(sess, model_path+str(epoch))
                all_prob = []

                for i in range(i_length):
                    print '%d/%d,test one batch'%(i,i_length)
                    temp_sentence_word = []
                    temp_sentence_pos1 = []
                    temp_sentence_pos2 = []                    
                    temp_sentence_y = []
                    temp_input = test_sentence_word[i*settings.batch_size:(i+1)*settings.batch_size]
                                                    
                    temp_sentence_word = test_sentence_word[i*settings.batch_size:(i+1)*settings.batch_size]
                    temp_sentence_pos1 = test_sentence_pos1[i*settings.batch_size:(i+1)*settings.batch_size]
                    temp_sentence_pos2 = test_sentence_pos2[i*settings.batch_size:(i+1)*settings.batch_size]
                    temp_sentence_y = test_y[i*settings.batch_size:(i+1)*settings.batch_size]

                    batch_sentence_word = np.array(temp_sentence_word)
                    batch_sentence_pos1 = np.array(temp_sentence_pos1)
                    batch_sentence_pos2 = np.array(temp_sentence_pos2)
                    batch_y = np.array(temp_sentence_y)

                    feed_dict = {}
                    feed_dict[m.input_word] = batch_sentence_word
                    feed_dict[m.input_pos1] = batch_sentence_pos1
                    feed_dict[m.input_pos2] = batch_sentence_pos2
                    feed_dict[m.input_y] = batch_y

                    loss_cur, acc, prob = sess.run([m.loss, m.accuracy, m.probility], feed_dict=feed_dict)
                    #_, step = sess.run([train_op, global_step], feed_dict=feed_dict)
                    for single_prob in prob:
                        #all_prob.append(single_prob)
                        #all_prob.append(single_prob[1:])
                        all_prob.append(single_prob[1:])
            
                ClassEst += np.multiply(all_prob, alpha)

            all_prob2 = []
            for index in range(test_length):
                all_prob2.append(np.argmax(softmax(ClassEst[index])))
            
            all_prob2 = np.reshape(np.array(all_prob2), -1)
            all_prob_length = len(all_prob2)
            #np.save('./output/all_prob.npy', all_prob2)
            np.save('./output/'+res_file, all_prob2)
            #print 'avg_pre_score'
            #print metrics.average_precision_score(all_ans[:all_prob_length], all_prob2)
            
            y_true=all_ans[:all_prob_length]
            y_pred=all_prob2

     

            #print 'PR curve area:' + str(average_precision)
            #print(classification_report(all_ans[:all_prob_length], all_prob2))
            
            
            order = np.argsort(-all_prob2)
            

            top100 = order[:100]
            correct_num_100 = 0.0
            for i in top100:
                if all_ans[i]==1:
                    correct_num_100 += 1.0
            
            
            top200 = order[:200]
            correct_num_200 = 0.0
            for i in top200:
                if all_ans[i]==1:
                    correct_num_200 += 1.0
            
            
            top300 = order[:300]
            correct_num_300 = 0.0
            for i in top300:
                if all_ans[i]==1:
                    correct_num_300 += 1.0
            
            top400 = order[:400]
            correct_num_400 = 0.0
            for i in top400:
                if all_ans[i]==1:
                    correct_num_400 += 1.0
            
            top500 = order[:500]
            correct_num_500 = 0.0
            for i in top500:
                if all_ans[i]==1:
                    correct_num_500 += 1.0

            top1000 = order[:1000]
            correct_num_1000 = 0.0
            for i in top1000:
                if all_ans[i]==1:
                    correct_num_1000 += 1.0


            print 'P@100\n'+str(correct_num_100/100.0)
            print 'P@200\n'+str(correct_num_200/200.0)
            print 'P@300\n'+str(correct_num_300/300.0)
            print 'P@400\n'+str(correct_num_400/400.0)
            print 'P@500\n'+str(correct_num_500/500.0)
            print 'P@1000\n'+str(correct_num_1000/1000.0)

            print '~~~Ending'


def main(_):
    #test_y = np.load('./data/test_y.npy')
    #test_sentence_word = np.load('./data/test_sentence_word.npy')
    #test_sentence_pos1 = np.load('./data/test_sentence_pos1.npy')
    #test_sentence_pos2 = np.load('./data/test_sentence_pos2.npy')
    #all_ans = np.load('./data/all_ans.npy')
    #def test_func(test_y_file, word_file, pos1_file, pos2_file, ans_file, res_file):
    test_func('test_y.npy', 'test_sentence_word.npy', 'test_sentence_pos1.npy', 'test_sentence_pos2.npy', 'all_ans.npy', 'all_prob.npy')
    #test_func('test_y_pone.npy', 'test_sentence_word_pone.npy', 'test_sentence_pos1_pone.npy', 'test_sentence_pos2_pone.npy', 'all_ans_pone.npy', 'all_prob_pone.npy')
    #test_func('test_y_ptwo.npy', 'test_sentence_word_ptwo.npy', 'test_sentence_pos1_ptwo.npy', 'test_sentence_pos2_ptwo.npy', 'all_ans_ptwo.npy', 'all_prob_ptwo.npy')

if __name__ == "__main__":
    tf.app.run()


