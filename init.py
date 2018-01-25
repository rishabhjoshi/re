import numpy as np
import os
import sys
import random
import string
from settings import Setting
#from sdp import tree

# for word_position_embedding
def train_word_position_embedding():
    settings = Setting()
    pos1 = np.random.uniform(-1.0, 1.0, size=[settings.pos_num, settings.pos_size])
    pos2 = np.random.uniform(-1.0, 1.0, size=[settings.pos_num, settings.pos_size])
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    pos1 = pos1.astype(np.float32)
    pos2 = pos2.astype(np.float32)
    np.save('./data/wordpos1.npy', pos1)
    np.save('./data/wordpos2.npy', pos2)


def find_index(x,y):
    flag = -1
    for i in range(len(y)):
        if x!= y[i]:
            continue
        else:
            return i
    return flag


def Embedding_Pos(x):
    if x < -60:
        return 0
    elif x >= 60 and x <= 60:
        return x+61
    else:
        return 122


def Read(filepath, f_sen, f_ans, relation2id, word2id, flag):
    
    fixlen = 70  #length of sentence is 70   can be other
    maxlen = 60  #max length of position embedding is 60  can be other
    
    f = open(filepath, 'r')

    random_flag = 0.0

    '''
    if flag == 1:
        #train data
        random_flag = 0.028
    else:
        #test data
        random_flag = 0.004
    '''
    # train: 
    # test: 
    

    line_num = 0
    
    while True:
        content = f.readline()
        
        if content == '':
            break
               
        line_num = line_num + 1
        #if line_num % 500 == 0:
        #    print line_num
        content = content.strip().split()

        entity1 = content[2]
        entity2 = content[3]
        relationID = 0 
        #print entity1, entity2, content

        if content[4] not in relation2id:
            relationID = relation2id['NA']
        else:
            relationID = relation2id[content[4]]

        tup = (entity1, entity2)
        label_tag = 0
        
        if tup not in f_sen:
            f_sen[tup] = []
            f_sen[tup].append([])
            y_id = relationID
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            f_ans[tup] = []
            f_ans[tup].append(label)
        else:
            y_id = relationID
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            temp = 0
            temp = find_index(label,f_ans[tup])
            if temp == -1:
                f_ans[tup].append(label)
                label_tag = len(f_ans[tup])-1
                f_sen[tup].append([])
            else:
                label_tag = temp

        sentence = content[5:-2]

        en1pos = 0
        en2pos = 0

        for i in range(len(sentence)):
            if sentence[i] == entity1:
                en1pos = i
            if sentence[i] == entity2:
                en2pos = i
        output = []


        # just to initialize the data structure 
        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = Embedding_Pos(i-en1pos) # 
            rel_e2 = Embedding_Pos(i-en2pos) # 
            output.append([word, rel_e1, rel_e2])

        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]
            output[i][0] = word

        '''
        index is the tup, which is tuple of two entities;
        value is the output, which is the list of [word, rel_e1, rel_e2]
        '''
        f_sen[tup][label_tag].append(output)

    f.close()


def Organized(newpath, sen, ans, sen_new, ans_new):

    f = open(newpath, 'w')
    temp = 0
    '''
    index is the tup, which is tuple of two entities;
    value is the output, which is the list of 
    [  [label1-sentence1,label1-sentence2,...],[label2-sentence1,lebel2-sentence2],...   ]
    '''
    for i in sen:
        if len(sen[i]) != len(ans[i]):
            print 'error'
        lenth = len(ans[i])
        for j in range(lenth):
            target=ans[i][j]
            for k in sen[i][j]:
                sen_new.append(k)
                ans_new.append(target)
                f.write(str(temp)+'\t'+i[0]+'\t'+i[1]+'\t'+str(np.argmax(target))+'\n')
                temp += 1
    print 'the size number of dataSet: %d' %temp
    f.close()

#for one-test, two-test, all-test
def Ptest(sen, ans, sen_new1, ans_new1, sen_new2, ans_new2):

    print 'ptest begin'
    f = open('./data/ptest1.npy', 'w')
    f2 = open('./data/ptest2.npy', 'w')
    temp1 = 0
    temp2 = 0

    for i in sen:
        lenth = len(ans[i])

        #one test
        temp = np.random.randint(lenth)
        target = ans[i][temp]
        temp2 = np.random.randint(len(sen[i][temp]))
        sentence = sen[i][temp][temp2]
        sen_new1.append(sentence)
        ans_new1.append(target)

        f.write(str(temp1)+'\t'+i[0]+'\t'+i[1]+'\t'+str(np.argmax(target))+'\n')
        temp1 += 1

        #two test
        temp = np.random.randint(lenth)
        target1 = ans[i][temp]
        temp2 = np.random.randint(len(sen[i][temp]))
        sentence1 = sen[i][temp][temp2]
        sen_new2.append(sentence1)
        ans_new2.append(target1)
        f2.write(str(temp2)+'\t'+i[0]+'\t'+i[1]+'\t'+str(np.argmax(target1))+'\t'+str(sentence1)+'\n')
        temp2 += 1
         
        
        random.seed()
        temp3 = np.random.randint(lenth)
        target2 = ans[i][temp3]
        random.seed()
        temp4 = np.random.randint(len(sen[i][temp3]))
        sentence2 = sen[i][temp3][temp4]
        if(lenth == 1 and len(sen[i][temp3])==1):
            continue
        else:
            while(temp1==temp3 and temp2==temp4):
                random.seed()
                temp3 = np.random.randint(lenth)
                target2 = ans[i][temp3]
                random.seed()
                temp4 = np.random.randint(len(sen[i][temp3]))
                sentence2 = sen[i][temp3][temp4]
                #print str(lenth)+'\t'+str(len(sen[i][temp3]))+'\t'+str(temp3)+'\t'+str(temp4) 
        #print 'still running'+str(temp1)
        sen_new2.append(sentence2)
        ans_new2.append(target2)
        f2.write(str(temp2)+'\t'+i[0]+'\t'+i[1]+'\t'+str(np.argmax(target2))+'\t'+str(sentence2)+'\n')
        temp2 += 1
        #-----------
    print 'ptest end'


def Seperate(sens, sens_word, sens_pos1, sens_pos2):
    for each_sen in sens:
        word_tmp = []
        pos1_tmp = []
        pos2_tmp = []
        for word in each_sen:
            word_tmp.append(word[0])
            pos1_tmp.append(word[1])
            pos2_tmp.append(word[2])
        sens_word.append(word_tmp)
        sens_pos1.append(pos1_tmp)
        sens_pos2.append(pos2_tmp)



if __name__=='__main__':
    
    reload(sys)
    sys.setdefaultencoding('utf8')
    
    train_word_position_embedding()

    f = open('./origin_data/vec.txt')
    print 'begin to read the word-embedding data'
    vec = [] #list
    word2id = {} #dict
    f.readline()
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)
        content_split = [i for i in content[1:]]
        vec.append(content_split)
    f.close()

    word2id['UNK']=len(word2id)  # 'UNK' is for the word not exist entity in vec.txt
    word2id['BLANK']=len(word2id)
    vec.append(np.random.normal(loc=0, scale=0.05, size=50))#scale is not sure
    vec.append(np.random.normal(loc=0, scale=0.05, size=50))#why two normal?
    vec = np.array(vec, dtype=np.float32)

    print 'reading relation to id'
    relation2id = {}
    f = open('./origin_data/relation2id.txt')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        relation2id[content[0]] = int (content[1])
    f.close()

    train_sen = {}
    train_ans = {}
    test_sen = {}
    test_ans = {}

    print 'reading train data'
    Read('./origin_data/train.txt', train_sen, train_ans, relation2id, word2id, 1)
    print 'reading test data'
    Read('./origin_data/test.txt', test_sen, test_ans, relation2id, word2id, 0)

    ####################################
    train_sen_new = []
    train_ans_new = []
    test_sen_new = []
    test_ans_new = []
    print 'organizing train data'
    Organized('./data/train_q&a.txt', train_sen, train_ans, train_sen_new, train_ans_new)
    Organized('./data/test_q&a.txt', test_sen, test_ans, test_sen_new, test_ans_new)

    ptest_sen_1 = []
    ptest_ans_1 = []
    ptest_sen_2 = []
    ptest_ans_2 = []
    Ptest(test_sen, test_ans, ptest_sen_1, ptest_ans_1, ptest_sen_2, ptest_ans_2)
        
    train_sen_new = np.array(train_sen_new)
    train_y = np.array(train_ans_new)
    test_sen_new = np.array(test_sen_new)
    test_y = np.array(test_ans_new)
    ptest_sen_1 = np.array(ptest_sen_1)
    ptest_ans_1 = np.array(ptest_ans_1)
    ptest_sen_2 = np.array(ptest_sen_2)
    ptest_ans_2 = np.array(ptest_ans_2)
    

    print 'seperating the dataset'
    train_word = []
    train_pos1 = []
    train_pos2 = []
    Seperate(train_sen_new, train_word, train_pos1, train_pos2)
    
    
    test_word = []
    test_pos1 = []
    test_pos2 = []
    Seperate(test_sen_new, test_word, test_pos1, test_pos2)
    
    ptest_word_1 = []
    ptest_pos1_1 = []
    ptest_pos2_1 = []
    Seperate(ptest_sen_1, ptest_word_1, ptest_pos1_1, ptest_pos2_1)

    ptest_word_2 = []
    ptest_pos1_2 = []
    ptest_pos2_2 = []
    Seperate(ptest_sen_2, ptest_word_2, ptest_pos1_2, ptest_pos2_2)

    
    # np.save("./data/all.npz", vec, train_x, train_y, test_x, test_y)
    np.save('./data/vec.npy', vec)

    np.save('./data/train_sentence_word.npy', train_word)
    np.save('./data/train_sentence_pos1.npy', train_pos1)
    np.save('./data/train_sentence_pos2.npy', train_pos2)
    np.save('./data/train_y.npy', train_y)
    
    np.save('./data/test_sentence_word.npy', test_word)
    np.save('./data/test_sentence_pos1.npy', test_pos1)
    np.save('./data/test_sentence_pos2.npy', test_pos2)
    np.save('./data/test_y.npy', test_y)
    
    np.save('./data/test_sentence_word_pone.npy', ptest_word_1)
    np.save('./data/test_sentence_pos1_pone.npy', ptest_pos1_1)
    np.save('./data/test_sentence_pos2_pone.npy', ptest_pos2_1)
    np.save('./data/test_y_pone.npy', ptest_ans_1)
    
    np.save('./data/test_sentence_word_ptwo.npy', ptest_word_2)
    np.save('./data/test_sentence_pos1_ptwo.npy', ptest_pos1_2)
    np.save('./data/test_sentence_pos2_ptwo.npy', ptest_pos2_2)
    np.save('./data/test_y_ptwo.npy', ptest_ans_2)
    
    
    
    #for testing
    eval_y = []
    for sample in test_y:
        eval_y.append(sample[1:]) 
        #eval_y.append(sample[1:])
        #eval_y.append(sample)
    eval_y = np.reshape(eval_y, -1)
    np.save('./data/all_ans.npy', eval_y)


    #for testing
    eval_y = []
    for sample in ptest_ans_1:
        eval_y.append(sample[1:])
        #eval_y.append(sample)
    eval_y = np.reshape(eval_y, -1)
    np.save('./data/all_ans_pone.npy', eval_y)


    #for testing
    eval_y = []
    for sample in ptest_ans_2:
        eval_y.append(sample[1:])
        #eval_y.append(sample)
    eval_y = np.reshape(eval_y, -1)
    np.save('./data/all_ans_ptwo.npy', eval_y)

