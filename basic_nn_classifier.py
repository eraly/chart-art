from lasagne import layers
from lasagne.nonlinearities import  softmax, rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from skimage import transform, io
import numpy as np
import cPickle
import pandas as pd
import os

class ImagePipeline(object):

    def __init__(self,image_dir='/home/ec2-user/scrape_images/image_data'):
        self.image_dir = image_dir
        self.image_data = []
        self.image_labels = []
        self.image_features = None
        

    def read_image_set(self,image_files_by_class):
        for a_class,its_images in image_files_by_class.iteritems():
            self.image_data.extend([io.imread(os.path.join(self.image_dir, image_file)) for image_file in its_images])
            image_labels.extend([int(a_class)] * len(its_images))

    def _resize(self,shape_to = (360, 360, 3)):
        resized_image_data = [transform.resize(an_image, shape_to) for an_image in self.image_data]
        self.image_data = resized_image_data

    def _vectorize(self,noPatch=True):
        if noPatch:
            row_tup = tuple(img_array.ravel()[np.newaxis, :]
                            for an_image in self.image_data for img_array in an_image)
            self.image_features = np.r_[row_tup]
        else:
            pass

    def process_pipeline(self):
        self._resize()
        self._vectorize()
        self.image_features = self.image_features.astype(np.float32)
        self.image_labels = self.image_labels.astype(np.int32)
        return self.image_features,self.image_labels

if __name__ == '__main__':
    #Vars
    subset_size = 375
    image_classes = ['genre-impressionism','genre-abstract']
    image_info_dict = {}

    print "Load in pickled dataframe"
    image_df = pd.read_pickle('../iPython_Notebooks/paintingspickle.p')

    for numeric_encoding,a_class in enumerate(image_classes):
        image_info_dict[str(numeric_encoding)] = list(image_df[image_df.category == a_class][:subset_size]['its_jpeg'])

    print 'Building Image Pipeline'
    image_pipeline = ImagePipeline()
    image_pipeline.read_image_set(image_info_dict)
    X,y = image_pipeline.process_pipeline()
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=13)
   
    
    print "Instantiating NN"
    nnet = NeuralNet(
              # Specify the layers
              layers=[('input', layers.InputLayer),
                      ('hidden1', layers.DenseLayer),
                      ('hidden2', layers.DenseLayer),
                      ('hidden3', layers.DenseLayer),
                      ('output', layers.DenseLayer)
                        ],

              # Input Layer
              input_shape=(None, X_scaled.shape[1]),

              # Hidden Layer 1
              hidden1_num_units=512,
              hidden1_nonlinearity=rectify,

              # Hidden Layer 2
              hidden2_num_units=512,
              hidden2_nonlinearity=rectify,

              # # Hidden Layer 3
              hidden3_num_units=512,
              hidden3_nonlinearity=rectify,

              # Output Layer
              output_num_units=3,
              output_nonlinearity=softmax,

              # Optimization
              update=nesterov_momentum,
              update_learning_rate=0.001,
              update_momentum=0.3,
              max_epochs=30,

              # Others,
              regression=False,
              verbose=1,
    )

    print "Training NN..."
    nnet.fit(X_train,y_train)

    print "Using trained model to predict"
    y_predictions = nnet.predict(X_test)
    print "f1 score:", f1_score(y_test, y_predictions, average='weighted')
    y_predictions = nnet.predict(X_test)

    print classification_report(y_test, y_predictions) 
    with open(r"basic_nn.pickle","wb") as output_file:
        cPickle.dump(nnet, output_file, protocol=cPickle.HIGHEST_PROTOCOL)
