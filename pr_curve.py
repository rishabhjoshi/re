
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Agg')

def file_read(fileName):
    f = open(fileName)
    line = f.readline()
    content = line.split()
    #print content
    return np.array(content)


#file_2017 = ['ATT_Mul_P.txt', 'ATT_Mul_R.txt']
#precision_2017 = file_read('./data/'+file_2017[0])
#recall_2017 = file_read('./data/'+file_2017[1])

plt.clf()
#filename = ['Hoffmann','MIMLRE','Mintz','CNN+ATT' ,'PCNN+ATT']
#color = ['red', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']

#for i in range(len(filename)):
#    print i
#    precision = np.load('./data/'+filename[i]+'_precision.npy')
#    recall  = np.load('./data/'+filename[i]+'_recall.npy')
#    print precision.shape
#    print recall.shape
#    plt.plot(recall, precision, color = color[i], lw=2, label=filename[i])

#plt.plot(recall_2017, precision_2017, color = 'pink', lw=2, label='Rank+ExATT')


y_true = np.load('./data/all_ans.npy')
y_scores = np.load('./output/all_prob_19.npy')
length = len(y_scores)
y_true = y_true[:length]
precision,recall,threshold = precision_recall_curve(y_true,y_scores)
plt.plot(recall[:], precision[:], lw=2, color='darkorange',label='Ada-LSTMs 1')
#average_precision = average_precision_score(y_true, y_scores)
y_scores = np.load('./output/all_prob_16_20.npy')
length = len(y_scores)
y_true = y_true[:length]
precision,recall,threshold = precision_recall_curve(y_true,y_scores)
plt.plot(recall[:], precision[:], lw=2, color='teal',label='Ada-LSTMs 5')

y_scores = np.load('./output/all_prob_11_20.npy')
length = len(y_scores)
y_true = y_true[:length]
precision,recall,threshold = precision_recall_curve(y_true,y_scores)
plt.plot(recall[:], precision[:], lw=2, color='yellow',label='Ada-LSTMs 10')

y_scores = np.load('./output/all_prob_11_30.npy')
length = len(y_scores)
y_true = y_true[:length]
precision,recall,threshold = precision_recall_curve(y_true,y_scores)
plt.plot(recall[:], precision[:], lw=2, color='navy',label='Ada-LSTMs 20')

y_scores = np.load('./output/all_prob_6_35.npy')
length = len(y_scores)
y_true = y_true[:length]
precision,recall,threshold = precision_recall_curve(y_true,y_scores)
plt.plot(recall[:], precision[:], lw=2, color='pink',label='Ada-LSTMs 30')

#y_true = np.load('./data/all_ans_from_att_baseline.npy')
y_scores = np.load('./output/all_prob_baseline.npy')
length = len(y_scores)
y_true = y_true[:length]
precision,recall,threshold = precision_recall_curve(y_true,y_scores)
plt.plot(recall[:], precision[:], lw=2, color='red',label='Ada-LSTMs without adaboost')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.55, 1.0])
plt.xlim([0.0, 0.45])

#plt.title('Precision-Recall')
#plt.title('Precision-Recall Area={0:0.2f}'.format(average_precision))
plt.legend(loc="upper right")
plt.grid(True)
#plt.savefig('iter_'+str(one_iter))
plt.savefig('result.pdf')



