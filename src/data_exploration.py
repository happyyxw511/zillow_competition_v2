import pandas as pd
data_path = '../data/'
properties_df = pd.read_csv(data_path + 'properties_2016.csv')
print properties_df.columns
train_df = pd.read_csv(data_path + 'train_2016.csv')
train_properties_df = train_df.merge(properties_df, on=['parcelid'])
train_properties_df.to_csv(data_path + 'train_properties.csv', index=False)
