import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random as rm
from captcha.image import ImageCaptcha
from PIL import Image
from tensorboard import summary

number=['0','1','2','3','4','5','6','7','8','9']
alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n',
          'o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET=['A','B','C','D','E','F','G','H','I','J','K','L','M','N',
          'O','P','Q','R','S','T','U','V','W','X','Y','Z']

def gray(image):
    if len(image.shape)>2:
        image=np.mean(image,-1)
    return image

def text_to_vec(text,capcha_len=4,capcha_list=number+alphabet+ALPHABET):
    text_len=len(text)
    if text_len>capcha_len:
        raise ValueError('验证码超过指定长度')
    v=np.zeros(capcha_len*len(capcha_list))
    for i in range(text_len):
        v[capcha_list.index(text[i])+i*len(capcha_list)]=1
    return v

def make_captcha_text(char_set=number+alphabet+ALPHABET,size=4):
    captcha_text=[]
    for i in range(size):
        captcha_text.append(rm.choice(char_set))
    return captcha_text

#生成验证码图像的函数
def make_text_and_image():
    image=ImageCaptcha()

    captcha_text=make_captcha_text()
    captcha_text=''.join(captcha_text)

    captcha=image.generate(captcha_text)

    captcha_image=Image.open(captcha)
    captcha_image=np.array(captcha_image)

    return captcha_text,captcha_image

def make_text_and_image_standard():
    while True:
        text,image=make_text_and_image()
        if image.shape==(60,160,3):
            return text,image

def get_next_batch(batch_size=128):
    batch_x=np.zeros([batch_size,image_h*image_w])
    batch_y=np.zeros([batch_size,max_captch*char_set_len])

    for i in range(batch_size):
        text,image=make_text_and_image_standard()
        image=gray(image)

        batch_x[i,:]=image.flatten()/255
        batch_y[i,:]=text_to_vec(text)

    return batch_x,batch_y

def make_cnn(w_alpha=0.01,b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, image_h, image_w, 1])

    wc1 = tf.get_variable(name='wc1', shape=[3, 3, 1, 32], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wc1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    bc1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 64], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wc2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    bc2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wc3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    bc3 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    wd1 = tf.get_variable(name='wd1', shape=[8 * 20 * 128, 1024], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wd1 = tf.Variable(w_alpha * tf.random_normal([7*20*128,1024]))
    bd1 = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
    dense = tf.nn.dropout(dense, keep_prob)

    wout = tf.get_variable('name', shape=[1024, 4 * char_set_len], dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer())
    # wout = tf.Variable(w_alpha * tf.random_normal([1024, max_captcha * char_set_len]))
    bout = tf.Variable(b_alpha * tf.random_normal([4 * char_set_len]))
    out = tf.add(tf.matmul(dense, wout), bout)

    return out


def train_cnn():
    out=make_cnn()
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels=Y))
    opti=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    pred=tf.reshape(out,[-1,max_captch,char_set_len])
    index_pred=tf.argmax(pred,2)
    index_Y=tf.argmax(tf.reshape(Y,[-1,max_captch,char_set_len]),2)
    corr=tf.equal(index_pred,index_Y)
    accu=tf.reduce_mean(tf.cast(corr,tf.float32))

    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step=0
        while True:
            batch_x,batch_y=get_next_batch(100)
            bx,real_loss=sess.run([opti,loss],feed_dict={X:batch_x,Y:batch_y,keep_prob:0.75})
            print('迭代次数:{},Cost:{}'.format(step,real_loss))

            if step%100==0 and step!=0:
                batch_x_test,batch_y_text=get_next_batch(100)
                sess.run(merged)
                acc=sess.run(accu,feed_dict={X:batch_x_test,Y:batch_y_text,keep_prob:1.0})
                print('已经迭代了{}次,准确率为{}%'.format(step,acc*100))

                if acc > 0.99:
                    saver.save(sess,'model/verify99.model')
                    break
                elif acc>0.80:
                    saver.save(sess,'model/verify80.model')
                elif acc>0.85:
                    saver.save(sess,'model/verify85.model')
                elif acc>0.90:
                    saver.save(sess,'model/verify90.model')
                elif acc>0.95:
                    saver.save(sess,'model/verify95.model')

            step+=1


def predict(image):
    out=make_cnn()

    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,'model/verify80.model')
        pred=tf.argmax(tf.reshape(out,[-1,4,char_set_len]),2)
        text_list=sess.run(pred,feed_dict={X:[image],keep_prob:1.0})
        text=text_list[0].tolist()
        return text;

if __name__=='__main__':

    train=1
    if train==0:
        text,image=make_text_and_image()
        # 图像大小
        image_h = 60
        image_w = 160
        max_captch = len(text)
        print('验证码文本最长字符数', max_captch)

        # 文本转换成向量
        char_set = number+alphabet+ALPHABET
        char_set_len = len(char_set)

        print('验证码图像的通道数:',image.shape)

        X=tf.placeholder(tf.float32,[None,image_h*image_w])
        Y=tf.placeholder(tf.float32,[None,max_captch*char_set_len])
        keep_prob=tf.placeholder(tf.float32)

        train_cnn()

    elif train==1:
        text, image = make_text_and_image()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        #ax.text(0.1,0.9,text,ha='center',va='center',transform=ax.transAxes)
        plt.imshow(image)

        # 图像大小
        image_h = 60
        image_w = 160
        max_captch = len(text)
        print('验证码文本最长字符数', max_captch)

        # 文本转换成向量
        char_set = number+alphabet+ALPHABET
        char_set_len = len(char_set)

        print('验证码图像的通道数:', image.shape)

        X = tf.compat.v1.placeholder(tf.float32, [None, image_h * image_w])
        Y = tf.compat.v1.placeholder(tf.float32, [None, max_captch * char_set_len])
        keep_prob = tf.compat.v1.placeholder(tf.float32)

        image=gray(image)
        image=image.flatten()/255

        pred_text=predict(image)
        result=[]
        for i in range(4):
            result.append(char_set[pred_text[i]])

        print('正确:{} 预测:{}'.format(text,result))

        plt.show()

    elif train==2:
        print('请输入图像的路径:')
        