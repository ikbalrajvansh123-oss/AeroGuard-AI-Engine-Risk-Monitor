from sklearn.preprocessing import StandardScaler
def scaling_data(train_df):
  feature_cols = [
      'op_setting_1','op_setting_2','op_setting_3',
      'sensor_2','sensor_3','sensor_4','sensor_7','sensor_8',
      'sensor_9','sensor_11','sensor_12','sensor_13',
      'sensor_14','sensor_15','sensor_17','sensor_20','sensor_21'
  ]
  scaler = StandardScaler()
  train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
  return train_df,scaler,feature_cols
