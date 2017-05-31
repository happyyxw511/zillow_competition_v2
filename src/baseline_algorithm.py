import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn
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


def construct_network(features):
    # Deep Neural Network
    prediction = layers.stack(features, layers.fully_connected, [10, 20, 1],
        activation_fn=tf.tanh)

def fit(prediction, train_y, train_x):
    target = tf.cast(train_y, tf.float64)
    prediction = tf.cast(prediction, tf.float64)
    loss = tf.reduce_mean(tf.square(prediction - target))
    train_op = layers.optimize_loss(loss,
        tf.contrib.framework.get_global_step(), optimizer='Adam', learning_rate=0.01)

    return prediction, loss, train_op


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


classifier.fit(input_fn=pandas_io.pandas_input_fn(X_train, y_train, num_epochs=10))
preds = list(classifier.predict(input_fn=pandas_io.pandas_input_fn(X_test, num_epochs=1), as_iterable=True))
print preds
print y_test.values
print(mean_squared_error(y_test.values, preds))
