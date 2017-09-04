
import tensorflow as tf
import numpy as np
import datetime
import time
import os
from settings import Setting
from model import GRU

from sklearn.metrics import average_precision_score


ada_weight = []
i = 0
ada_length = 50
delta = ada_length

@tf.RegisterGradient("AdaGrad")
def _ada_grad(op, grad):
    return grad * ada_weigth[i] * delta


       

def main(_):
    print 'reading word-embedding'
    wordembedding = np.load('./data/vec.npy')
    wordpos1 = np.load('./data/wordpos1.npy')
    wordpos2 = np.load('./data/wordpos2.npy')

    print 'reading training data'
    train_y = np.load('./data/train_y.npy')
    train_sentence_word = np.load('./data/train_sentence_word.npy')
    train_sentence_pos1 = np.load('./data/train_sentence_pos1.npy')
    train_sentence_pos2 = np.load('./data/train_sentence_pos2.npy')

    precious_record = open('./precious_record.txt', 'w')


    save_path = './model/ATT-MODEL'
    
    settings = Setting()
    settings.num_classes = len(train_y[0])
    settings.voc_size = len(wordembedding)
        
    batch_size = settings.batch_size

    with tf.Graph().as_default():
       sess = tf.Session()
       with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            print "model training begin"
            
            
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = GRU(is_training=True, word_embeddings=wordembedding, word_pos1=wordpos1, word_pos2=wordpos2, settings=settings)
            

            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("logs/", sess.graph) # tensorflow >=0.12



            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(settings.learning_rate)
            
            
            all_alpha = []

            i_length = int(len(train_sentence_word)/float(settings.batch_size))
            train_length = i_length * settings.batch_size
            global ada_length
            ada_length = i_length
            global ada_weight
            ada_weight = np.array(np.ones(ada_length) / ada_length)
            global i
            i = 0

            ggClassEst = np.zeros((train_length, 1))
            params = tf.trainable_variables()
            g = tf.get_default_graph()
            with g.gradient_override_map({"Ada": "AdaGrad"}):
                m.loss = tf.identity(m.loss, name="Ada")
            train_op = optimizer.minimize(m.loss, global_step=global_step)
            
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)
            print 'Starting Learning'
            

            feed_dict_list = []

            for epoch in range(settings.num_epochs):
                rand_order = range(i_length)
                #np.random.shuffle(rand_order)
                ii = 0
                all_prob = []
                all_err = []
                all_comp = []
                batch_err = []
                batch_comp = []
                for index in range(10):
                    print ada_weight[index]

                for i in rand_order:
                    print('epoch-%d, %d/%d (%d):run one batch'%(epoch, ii, i_length, i))
                    ii += 1
                    temp_sentence_word = []
                    temp_sentence_pos1 = []
                    temp_sentence_pos2 = []                    
                    temp_sentence_y = []
                    
                    temp_sentence_word = train_sentence_word[i*settings.batch_size:(i+1)*settings.batch_size]
                    temp_sentence_pos1 = train_sentence_pos1[i*settings.batch_size:(i+1)*settings.batch_size]
                    temp_sentence_pos2 = train_sentence_pos2[i*settings.batch_size:(i+1)*settings.batch_size]
                    temp_sentence_y = train_y[i*settings.batch_size:(i+1)*settings.batch_size]
                    
                    batch_sentence_word = np.array(temp_sentence_word)
                    batch_sentence_pos1 = np.array(temp_sentence_pos1)
                    batch_sentence_pos2 = np.array(temp_sentence_pos2)
                    batch_y = np.array(temp_sentence_y)
                    
                    feed_dict = {}
                    feed_dict[m.input_word] = batch_sentence_word
                    feed_dict[m.input_pos1] = batch_sentence_pos1
                    feed_dict[m.input_pos2] = batch_sentence_pos2
                    feed_dict[m.input_y] = batch_y
                    
                    #temp, step, loss_cur, acc = sess.run([train_op, global_step, m.loss, m.accuracy], feed_dict=feed_dict)
                    temp, step, loss_cur, acc, prob, comp, err = sess.run([train_op, global_step, m.loss, m.accuracy, m.probility, m.comparison, m.difference], feed_dict=feed_dict)

                    time_str = datetime.datetime.now().isoformat()
                    

                    for single_prob in prob:
                        all_prob.append(single_prob)
                    for single_err in err:
                        all_err.append(single_err)
                    for single_comp in comp:
                        all_comp.append(single_comp)
                    batch_err.append(np.sum(err)/settings.batch_size)
                    batch_comp.append(np.sum(comp)/settings.batch_size)
                
                    if step % 10 ==0:
                        print("  {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_cur, acc))
                        rs = sess.run(merged, feed_dict=feed_dict)
                        writer.add_summary(rs, step)
                
                #all_prob = np.reshape(np.array(all_prob), -1)
                #all_prob_length = len(all_prob)
                #average_precision = average_precision_score(all_ans[:all_prob_length], all_prob)
                #print 'PR curve area:' + str(average_precision)
                
                #errorMisClassified = sum([np.exp(all_err[index]) for index in range(train_length) if not all_comp[index]])

                #errorWellClassified = sum([np.exp(all_err[index]) for index in range(train_length) if all_comp[index]])
                errorMisClassified = sum([np.exp(batch_err[index]) for index in range(ada_length) if batch_comp[index]<0.5 ])
                errorWellClassified = sum([np.exp(batch_err[index]) for index in range(ada_length) if batch_comp[index]>=0.5 ])
                
                alpha = 0.5 * np.log(errorWellClassified / errorMisClassified)
                print 'alpha:', alpha
                all_alpha.append(alpha)
                for index in range(ada_length):
                    if batch_comp[index] >= 0.5:
                        ada_weight[index] *= np.exp(-alpha + batch_err[index])
                    else:
                        ada_weight[index] *= np.exp(alpha + batch_err[index])
                max_ada_weight = max(ada_weight)
                global delta
                delta = 1.0 / max_ada_weight
                print 'delta:', delta
                ada_weight = np.dot(ada_weight, 1./sum(ada_weight))
                precious_record.writelines("epoch: " + str(epoch) +" alpha: "+str(alpha)+"\n")
                saver.save(sess, save_path+str(epoch))


            np.save('./data/all_alpha.npy', all_alpha)
            saver.save(sess, save_path+"_final")

            print 'Trainning finished, Model being saved'
            #current_step = tf.train.global_step(sess, global_step)
            #path = saver.save(sess, save_path+'ATT_Model', global_step=current_step)
            #saver.save(sess, save_path)
    print '-------end------'



# tf.nn.softmax(tf.matmul(x,W)+b)
# tf.nn.dropout
if __name__ == "__main__":
    tf.app.run()


