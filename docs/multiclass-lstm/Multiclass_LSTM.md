
<center><h1>LSTM for k-hot Encoded Multiclassification</h1></center>

In this Notebook, we will Train a LSTM using K Hot Encoded Multi Classification. We will build a Neural Network architecture, and use the Featureized results from previous Notebooks as Input data.

#### Imports
To start, we will make the necessary imports. The important libraries being used here are Keras and Tensorflow. get_labels is a Class we wrote to obtain Labels from our dataset.


```python
import numpy as np

from get_labels import get_labels
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Dropout
from keras.layers import LSTM
from keras.optimizers import *;

from keras.callbacks import ModelCheckpoint, Callback

import urllib2;
import json;
```

    Using TensorFlow backend.


#### Progress Tracking
We will use the following Method to post messsage on our Slack channel. This will allow us to obtain Status updates as our Model runs.


```python
'''
Helper function to send notifications on Slack via Post CURL.
'''
def notify_slack(message):
    url = 'https://hooks.slack.com/services/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';
    post_fields = json.dumps({'channel' : '#random', 'username': 'webhookbot', 'text': message});

    request = urllib2.Request(url, post_fields);
    response = urllib2.urlopen(request)
    read_val = response.read()

```

#### Obtaining the Lables

The next step is to obtain the Labels for our data. We will store these in a new variable, and will have the format of being K-Hot encoded. So, for instance, if a Patient has ICD-9 codes 4, 5, and 9 assigned to him/her, our of Total 100 possible values, that patient will be given a 100-length long Zero-vector, with positions 4, 5, and 9 activated and set to 1. This vectorization will be done for each Clinical note, so for our ~2,000,000 notes, there will be ~2 Millions vectors returned.


```python
labels = get_labels()
notify_slack('Got labels');
```

    /home/ubuntu/.local/lib/python2.7/site-packages/IPython/core/interactiveshell.py:2881: DtypeWarning: Columns (4,5) have mixed types. Specify dtype option on import or set low_memory=False.
      exec(code_obj, self.user_global_ns, self.user_ns)


#### Resizing the Lables

To use them with our model, the Labels have to be a reshaped to a [1, \*] Vector.


```python
labels_array = np.array([x for x in labels])
labels_reshaped = labels_array.reshape(1851243, 1, 1070)
notify_slack('Labeles Reshaped');
```

#### Obtaining the Training Data
We have already processed the features into Training data, which have been stored as a JobLib file. That data will be loaded next.


```python
train_x = joblib.load("/mnt/cleaned_tfidf_reduced_420_morning")
notify_slack('Loaded Train X');
```

#### Re-sizing the Training Data

Similar to Labels, we process the Training data as well to have the reshaped.


```python
print (train_x.shape)
train_x_reshaped = train_x.reshape(1851243,1,1000)
print (train_x_reshaped.shape)
```

    (1851243, 1000)
    (1851243, 1, 1000)


#### Creating a Train-Test Split

Now we create a Train Test split on our data. Doing this will give us a 80% / 20% split, where we can use the 80% split for training and 20% for testing.


```python
x_train, x_test, y_train, y_test = train_test_split(train_x_reshaped, labels_reshaped, test_size=0.20)
notify_slack('Obtained Train Test Split');
```

There are some variables which take up a lot of memory. We will delete those to free up the consumed RAM and ease computation.


```python
del train_x
del labels_array
del labels_reshaped
```


```python
print ("\n New shapes:")
print("x_train shape", x_train.shape)
print("x_test shape", x_test.shape)

print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)
```

    New shapes:
    ('x_train shape', (1480994, 1, 1000))
    ('x_test shape', (370249, 1, 1000))
    ('y_train shape', (1480994, 1, 1070))
    ('y_test shape', (370249, 1, 1070))


#### Create Multiclass Loss Function

For our LSTM, we will use a Multi-Class Loss function.


```python
'''
A custom Loss function for Multi-Class Prediction.
'''
def multiclass_loss(y_true, y_pred):
    EPS = 1e-5
    y_pred = K.clip(y_pred, EPS, 1 - EPS)
    return -K.mean((1 - y_true) * K.log(1 - y_pred) + y_true * K.log(y_pred))
```

#### Instantiate Parameters for LSTM


```python
shape = x_train.shape[2]
num_classes = 1070
```

#### Create and Compile LSTM

Now, we will create the LSTM. The first thing we need to do is to create a Callback function for our model. This class will be added to a list of Model checkpoints, which will run the following items: It will save the losses in a List, increment the Total number of epochs completed, and Notify Slack on the progress of each Epoch.


```python
'''
A Class that acts as a Callback between each Epoch, used to monitor progress.
'''
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.num = 0;

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.num = self.num + 1;
        notify_slack('Finished epoch ' + str(self.num) + ' with ' + str(logs));
```

Next, we define the Model architecture. It is best described by the following image:

<html>
<center><img src='LSTM Model.png' width=500px></img></center>
</html>

We use a LSTM Layer with 32 Neurons, followed by a Dropout and a Dense layer. The optimizer used is the Adam optimizer, with deault values. After trying different architectures, we settled on this because of the high ROC-AUC returned by it.


```python
'''
Returns a Model Object instantiated with a LSTM Layer and a Dense Layer, along with an Adam optimizer.
'''
def create_model(shape,num_classes):
    print (type(shape), type(num_classes))
    print (shape, num_classes) # (None,shape)
    
    model = Sequential()
    
    '''
    model.add(LSTM(output_dim=128, input_shape=(None, shape), return_sequences=True));
    '''
    
    model.add(LSTM(output_dim=32, input_shape=(None, shape), return_sequences=True));
    model.add(Dropout(rate=0.5));
    model.add(Dense(num_classes, kernel_initializer='normal', activation='sigmoid'));
    
    '''
    opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=0.1, decay=0.05);
    '''
    
    filepath = 'model_checkpoint'
    history = LossHistory()
    
    model.compile(loss=multiclass_loss, optimizer='adam', metrics=['accuracy', 'mse', 'mae']);
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    callbacks_list = [checkpoint, history];

    return model, callbacks_list
```

#### Train LSTM

The next step is to actual train the LSTM. We use 50 epochs, on a Batch size of 512. 


```python
model, callbacks_list = create_model(shape, num_classes)
notify_slack('Obtained LSTM Model and starting training');
history = model.fit(x_train, y_train,
              batch_size=512, epochs=50,
              verbose = 1, callbacks=callbacks_list, validation_split=0.125)
notify_slack('Completed LSTM Model on 50 epochs');
```

    (<type 'int'>, <type 'int'>)
    (1000, 1070)


    /home/ubuntu/.local/lib/python2.7/site-packages/ipykernel_launcher.py:8: UserWarning: Update your `LSTM` call to the Keras 2 API: `LSTM(units=32, return_sequences=True, input_shape=(None, 100...)`
      


    Train on 1295869 samples, validate on 185125 samples
    Epoch 1/3
    1295744/1295869 [============================>.] - ETA: 0s - loss: 0.0534 - acc: 0.0066 - mean_squared_error: 0.0131 - mean_absolute_error: 0.0304Epoch 00000: val_acc improved from -inf to 0.00845, saving model to model_checkpoint
    {'acc': 0.0066395600172548305, 'loss': 0.053389187245908326, 'mean_absolute_error': 0.030402256992436344, 'val_mean_squared_error': 0.0091845857461462909, 'val_mean_absolute_error': 0.018135590655187109, 'val_acc': 0.0084537474679270766, 'mean_squared_error': 0.013141528600295455, 'val_loss': 0.037410789584626548}
    1295869/1295869 [==============================] - 117s - loss: 0.0534 - acc: 0.0066 - mean_squared_error: 0.0131 - mean_absolute_error: 0.0304 - val_loss: 0.0374 - val_acc: 0.0085 - val_mean_squared_error: 0.0092 - val_mean_absolute_error: 0.0181
    Epoch 2/3
     139904/1295869 [==>...........................] - ETA: 99s - loss: 0.0386 - acc: 0.0081 - mean_squared_error: 0.0093 - mean_absolute_error: 0.0191


#### Save LSTM Model

We will save the Model after it is completed, so that it can be loaded in for prediction later.


```python
from keras.models import load_model
model.save('khot_LSTM_216.h5') 
notify_slack('Saved LSTM Model');
```
