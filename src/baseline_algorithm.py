import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.learn.python.learn.learn_io import pandas_io
data_path = '../data/'
train_properties = pd.read_csv(data_path + 'train_properties.csv', index_col=False)
y = train_properties.pop('logerror')

categorical_vars = [
    'architecturalstyletypeid',
    'airconditioningtypeid',
    'buildingclasstypeid',
    'buildingqualitytypeid',
    'decktypeid',
    'heatingorsystemtypeid',
    'pooltypeid10',
    'pooltypeid2',
    'pooltypeid7',
    'propertycountylandusecode',
    'propertylandusetypeid',
    'propertyzoningdesc',
    'regionidcity',
    'regionidcounty',
    'regionidneighborhood',
    'regionidzip',
    'storytypeid',
    'typeconstructiontypeid',
    'taxdelinquencyflag',
    'hashottuborspa',
    'fips'
]

continuous_vars = [
    'basementsqft',
    'bathroomcnt',
    'bedroomcnt',
    'calculatedbathnbr',
    'finishedfloor1squarefeet',
    'calculatedfinishedsquarefeet',
    'finishedsquarefeet12',
    'finishedsquarefeet13',
    'finishedsquarefeet15',
    'finishedsquarefeet50',
    'finishedsquarefeet6',
    'fireplacecnt',
    'fullbathcnt',
    'garagecarcnt',
    'garagetotalsqft',
    'lotsizesquarefeet',
    'poolcnt',
    'poolsizesum',
    'unitcnt',
    'yardbuildingsqft17',
    'yardbuildingsqft26',
    'yearbuilt',
    'numberofstories',
    'structuretaxvaluedollarcnt',
    'taxvaluedollarcnt',
    'assessmentyear',
    'landtaxvaluedollarcnt',
    'taxamount'
]

meta_columns = [
    'transactiondate',
    'latitude',
    'longitude'
]

CATEGORICAL_EMBED_SIZE = 10 # Note, you can customize this per variable.


def preprocess_pandas(features_df):
    # Organize continues features.
    final_features = [tf.expand_dims(tf.cast(features_df[var], tf.float32), 1) for var in continuous_vars]
    # Embed categorical variables into distributed representation.
    for var in categorical_vars:
        feature = layers.embed_sequence(
            features_df[var + '_ids'], vocab_size=len(categorical_var_encoders[var].classes_),
            embed_dim=CATEGORICAL_EMBED_SIZE, scope=var)
        final_features.append(feature)
    # Concatenate all features into one vector.
    features = tf.concat(final_features, 1)
    return features


def _mlp_layer(name, input_size, output_size, input):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=[input_size, output_size], dtype=tf.float64, initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable('bias', shape=[output_size], dtype=tf.float64, initializer=tf.zeros_initializer())
        return tf.multiply(input, weight) + bias


def construct_network(input, layers):
    """
    layers are a list of number, the length is the depth of the layer and the number is the layer's size
    :param input:
    :param layers:
    :return:
    """
    # layer one
    with tf.variable_scope('feedforward_network'):
        input_shape = input.get_shape().as_list()
        layer_name = 'layer_{}'
        input_size = input_shape[1]
        for index, output_size in enumerate(layers):
            name = layer_name.format(index)
            layer = _mlp_layer(name, input_size, output_size, input)
            input = layer
            input_size = output_size
        return input



def construct_train_graph(network, y):
    loss = tf.reduce_mean(tf.square(network - y))
    optimizer = tf.train.AdamOptimizer()
    return optimizer.minimize(loss)

def train_loop(input_X, input_Y, optimizer_op, batch_size, num_epoch, num_items):
    for epoch in xrange(num_epoch):
        num_iters = np.floor(num_items / batch_size)

X = train_properties[categorical_vars + continuous_vars].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.copy()
X_test = X_test.copy()
categorical_var_encoders = {}
for var in categorical_vars:
    print var
    le = LabelEncoder().fit(X[var])
    X_train[var + '_ids'] = le.transform(X_train[var])
    X_test[var + '_ids'] = le.transform(X_test[var])
    X_train.pop(var)
    X_test.pop(var)
    categorical_var_encoders[var] = le


input_X = tf.placeholder(dtype=tf.float64)
input_Y = tf.placeholder(dtype=tf.float64)
layers = [30, 20, 30]
network = construct_network(input, layers)
optimizer_po = construct_train_graph(network, input_Y)
with tf.Session() as sess:
    sess.run(optimizer_po, feed_dict={
        input_X: X_train,
        input_Y: y_train
    })


classifier.fit(input_fn=pandas_io.pandas_input_fn(X_train, y_train, num_epochs=10))
preds = list(classifier.predict(input_fn=pandas_io.pandas_input_fn(X_test, num_epochs=1), as_iterable=True))
print preds
print y_test.values
print(mean_squared_error(y_test.values, preds))
