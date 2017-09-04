import tensorflow as tf
#from sdp import tree


def Attention_layer(inputs):

    attention_size = 50
    '''
    Attention mechnism layer
    :param inputs:outputs of RNN/Bi-RNN layer(not final state)
    :param attention_size: linear size of attention weights
    :param output of the passed RNN/Bi-RNN reduced with attention layer
    
    input: [total_num, num_steps, hidden_size]    
    output: [batch_size, hidden_size]
    '''

    sequence_length = inputs.get_shape()[1].value
    hidden_size = inputs.get_shape()[2].value

    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1,-1]))
    
    #vu [total_num*num_steps, attention_size]
    vu = tf.matmul(v, tf.reshape(u_omega, [-1,1]))

    #exps [total_num, num_steps]
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])

    #normalization
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    #output level, [total_num, num_steps, hidden_size] * [total_num, num_steps, 1]
    # [total_num, num_steps, hidden_size]
    output = tf.reduce_sum(inputs * tf.reshape(alphas,[-1, sequence_length, 1]),1)

    return output
    
def Syn_attention_layer(inputs, sdp):

    attention_size = 50
    sequence_length = inputs.get_shape()[1].value
    hidden_size = inputs.get_shape()[2].value
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1,-1]))
    #vu [total_num*num_steps, attention_size]
    vu = tf.matmul(v, tf.reshape(u_omega, [-1,1]))
    #exps [total_num, num_steps]
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    #normalization
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
    
    #------------Gaussian------------
    #settings: windows_size
    D = 3.0

    W_g = tf.Variable(tf.random_normal([hidden_size, 1], stddev=0.1))
    #[total_num*num_steps, 1]
    gs_pos = tf.nn.sigmoid(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_g))
    p_t = sequence_length * tf.reshape(gs_pos, [-1, sequence_length])

    sig = D / 2
    
    #sdp:  [batch_size, num_steps]
    gs = tf.exp(- (sdp - p_t) * (sdp - p_t) / (2 * sig * sig))

    #output level, [total_num, num_steps, hidden_size] * [total_num, num_steps, 1]
    # reduce_sum, 1: [total_num, num_steps, hidden_size]
    output = tf.reduce_sum(inputs * \
            tf.reshape(alphas,[-1, sequence_length, 1]) * \
            tf.reshape(gs, [-1, sequence_length, 1])
            ,1)
    

    return output

