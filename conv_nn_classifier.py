import theano
import glob
from theano.sandbox.cuda import dnn
from lasagne import layers
import theano.tensor as T
from lasagne.updates import sgd, momentum, adagrad, nesterov_momentum
from lasagne.objectives import binary_crossentropy
from lasagne.nonlinearities import softmax,rectify,sigmoid
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
'''
note: Conv2DDNNLayer only works on CudaDNN enabled systems
additionally Conv2DLayer can return different sizes from Conv2DDNNLayer
in some situations
I would recommend using Conv2DLayer 
'''
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer 
#from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer

from nolearn.lasagne import NeuralNet, BatchIterator
from sklearn.metrics import hamming_loss

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from skimage import transform, io
import numpy as np
import cPickle
import pandas as pd
import os
import datetime

class ImagePipeline(object):

    def __init__(self,image_dir='/home/ubuntu/scrape_images/image_data',shape_to =(320,320,3)):
        self.image_dir = image_dir
        self.shape_to = shape_to
        self.image_data = []
        self.image_labels = []
        self.current_image = None
        self.current_label = None
        

    def process_pipeline(self,image_files_by_class):
        for a_class,its_images in image_files_by_class.iteritems():
            for image_file in its_images:
                self.process_image(image_file,a_class)
        self.encode_labels()
        self.correct_casting()
        return self.image_data,self.image_labels

    def _load(self,image_file):
        image_id = image_file.split('/')[-1].split('__')[0]
        image_file = glob.glob(self.image_dir+"/"+str(image_id)+'__*jpg')[0]
        self.current_image = io.imread(image_file)

    def _resize(self):
        self.current_image = transform.resize(self.current_image, self.shape_to)

    def _reshape(self):
        self.current_image = np.swapaxes(np.swapaxes(self.current_image, 1, 2), 0, 1)
            
    def _append_image(self,a_class):
        self.image_data.append(self.current_image)
        self.image_labels.append(a_class)

    def process_image(self,image_file,a_class):
    
        self._load(image_file)
        if len(self.current_image.shape) == 3:
            self._resize()
            self._reshape()
            self._append_image(a_class)
    
    def correct_casting(self):
        #one_ht_enc = OneHotEncoder()
        #one_ht_enc.fit([[0],[1]]) 
        #print "BEFORE one hot transformation"
        #print self.image_labels[-1]
        #self.image_labels = one_ht_enc.transform(self.image_labels).toarray()
        #print "One hot transformation"
        #print self.image_labels[-1]
        self.image_labels = np.array(self.image_labels).astype(np.float32)
        #print "After conversion to numpy array"
        #print self.image_labels[-1]
        self.image_data = np.array(self.image_data).astype(np.float32)

    def encode_labels(self):
        self.image_labels = [set(x.split(", ")) for x in self.image_labels]
        self.image_labels = MultiLabelBinarizer().fit_transform(self.image_labels)

    def free_memory(self):
        self.current_image = None
        self.current_label = None

# custom loss: multi label cross entropy
def multilabel_objective(predictions, targets):
    epsilon = np.float32(1.0e-6)
    one = np.float32(1.0)
    pred = T.clip(predictions, epsilon, one - epsilon)
    return -T.sum(4*targets * T.log(pred) + (one - targets) * T.log(one - pred), axis=1)

if __name__ == '__main__':
    from scrape_for_images import main_run
    #Vars
    subset_size = 200
    image_info_dict = {}
    scale_size = 60
    image_shape = (None,3,scale_size,scale_size)

    #image_classes = ['genre-impressionism','genre-abstract']
    #styles = ['genre-abstract', 'classical-artwork', 'genre-expressionism',\
    #          'genre-impressionism','minimalism-artwork','modern-artwork', \
    #          'genre-pop culture','primitive-artwork', 'realism-artwork', \
    #          'street-art-artwork', 'genre-surrealism', 'vintage-artwork']


    #if True:
    if False:

        print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "Load in pickled dataframe"
        image_df = pd.read_pickle('paintingspickle.p')
        
        for i,a_class in enumerate(list(image_df.category.values)):
            #image_info_dict[a_class] = list(image_df[image_df.category == a_class][:subset_size]['its_jpeg'])
            image_info_dict[a_class] = list(image_df[image_df.category == a_class][:]['its_jpeg'])
        
        print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        
        print 'Building Image Pipeline'
        image_pipeline = ImagePipeline(shape_to=(scale_size,scale_size,3))
        X,y = image_pipeline.process_pipeline(image_info_dict)
        
        print "Saving X and y"
        X.dump("saved_X")
        y.dump("saved_y")
        print "Done building pipeline. Train,test split"
        print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

    else:
        X = np.load("saved_X")
        y = np.load("saved_y")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=13)
    #skf = StratifiedKFold([str(x) for x in y], n_folds=2,shuffle=True)
    #for train_index, test_index in skf:
    #   print("TRAIN:", train_index, "TEST:", test_index)
    #   X_train, X_test = X[train_index], X[test_index]
    #   y_train, y_test = y[train_index], y[test_index]


    print "Here is y train"
    print y[-1]
    
    print "Instantiating NN"
    print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    nnet = NeuralNet(
        layers=[
        ('input', InputLayer),
        ('conv10', layers.Conv2DLayer),
        ('conv11', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv20', layers.Conv2DLayer),
        ('conv21', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('hidden1', DenseLayer),
        ('dropout1', DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('dropout2', DropoutLayer),
        ('hidden3', DenseLayer),
        ('output', DenseLayer),
        ],
        input_shape=image_shape,
        conv10_num_filters=64, 
        conv10_filter_size=(16, 16), 
        conv10_nonlinearity=rectify,
        conv11_num_filters=64, 
        conv11_filter_size=(12, 12), 
        conv11_nonlinearity=rectify,
        #conv11_border_mode="valid",
        pool1_pool_size=(2, 2),
        conv20_num_filters=40, 
        conv20_filter_size=(5, 5), 
        pool2_pool_size=(2, 2),
        conv21_num_filters=20, 
        conv21_filter_size=(5, 5), 
        hidden1_num_units = 1024,
        hidden1_nonlinearity=rectify,
        dropout1_p = 0.5,
        hidden2_num_units = 512,
        hidden2_nonlinearity=rectify,
        dropout2_p = 0.5,
        hidden3_num_units=128,
        hidden3_nonlinearity=rectify,
        output_num_units = y_train.shape[1],
        output_nonlinearity = sigmoid,
        #output_nonlinearity = softmax,
        update=nesterov_momentum,
        #update=adagrad,
        #update=sgd,
        update_learning_rate=0.0001,
        update_momentum=0.95,
        regression = True,
        #objective_loss_function=binary_crossentropy,
        objective_loss_function=multilabel_objective,
        custom_score=("validation score", lambda x, y: np.mean(np.abs(x - y))),
        max_epochs= 100,
        verbose=1,
        )
    print "Training NN..."
    print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    X_offset = np.mean(X_train, axis = 0)
    nnet.fit(X_train-X_offset,y_train)

    print "Using trained model to predict"
    print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    y_predictions = nnet.predict(X_test-X_offset)

    print datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    score = 0
    for i,j in zip(y_test,y_predictions):
        temp = []
        for a in j:
            if a > 0.5:
                temp.append(1.)
            else:
                if a == max(j):
                    temp.append(1.)
                else:
                    temp.append(0.)
        if list(i) == temp:
            score += 1
        else:
            print i,j
    print "My accuracy score is:",score," right of",y_predictions.shape[0]
    print "How close am I?",np.mean(np.abs(y_test - y_predictions))

    #with open(r"basic_nn.pickle","wb") as output_file:
    #    cPickle.dump(nnet, output_file, protocol=cPickle.HIGHEST_PROTOCOL)
