#coding:utf-8

import numpy as np
import tornado.web
import tornado.ioloop
import tornado.httpserver
import tornado.options
import tornado.websocket
from tornado.options import options, define
from tornado.web import RequestHandler, MissingArgumentError
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_cnn as mnist_interence
import matplotlib.pyplot as plt
import base64
import mnist_train as mnist_train

define("port", default=8000, type=int, help="run server on the given port.")

class NumHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print("websocket start")

    def on_message(self,message):
        #print(message)
        '''b64str=message.split(";base64,")[1] #取出base64编码
        print(b64str)
        b64str=base64.b64decode(b64str)
        print(b64str)
        urlsafe64=base64.urlsafe_b64encode(b64str)  #转成websafe的
        base64_tensor=tf.convert_to_tensor(urlsafe64,dtype=tf.string)
        img_str=tf.decode_base64(base64_tensor)
        img=tf.image.decode_image(img_str,channels=4)
        with tf.Session() as sess:
            img_value=sess.run([img])[0]
            print(img_value)
            greyimg=np.zeros((28,28),dtype=int)
            for i in range(28):
                for j in range(28):
                    greyimg[i][j]=img_value[i][j][0]    #取RGB任意一个通道的值作为灰度值
            plt.imshow(greyimg,cmap='gray')
            plt.show()
            greyimg=greyimg.reshape((1,28,28))'''
            #print(greyimg)
        img=list(message)
        imgdata=np.array(img)
        imgdata=imgdata.reshape((28,28,4))
        #print(imgdata)
        greyimg=np.zeros((28,28),dtype=int)
        for i in range(28):
            for j in range(28):
                greyimg[i][j]=imgdata[i][j][0]  #取RGB任意一个通道的值作为灰度值
            #print(greyimg)
        #plt.imshow(greyimg,cmap='gray')
        #plt.show()
        greyimg=np.array(greyimg,dtype="float32")/255  #转float32
        
        greyimg=greyimg.reshape((1,28,28,1))
        tf.reset_default_graph()   #Variable layer1-conv/w already exists, disallowed
        #定义输入的张量
        x = tf.placeholder(tf.float32, shape=[1,
                                              mnist_interence.IMAGE_SIZE,
                                              mnist_interence.IMAGE_SIZE,
                                              mnist_interence.NUM_CHANNEL], name='x-input')

        val_feed = {x: greyimg}
        y = mnist_interence.interence(x,False,None)    #定义输出的张量
        saver=tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                result=sess.run(y,feed_dict=val_feed)         #识别结果返回result
                print(result)              
                self.write_message(str(result.argmax()))     #返回最大一个的下标,即数字


class IndexHandler(RequestHandler):    
    def get(self):
        self.render('index.html',title='输入数字')

 
if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application([
        (r"/", IndexHandler),
        (r"/num",NumHandler),
    ])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.current().start()
