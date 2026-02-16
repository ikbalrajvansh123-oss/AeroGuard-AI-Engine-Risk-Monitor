import numpy as np

def create_sequences(df,feature_cols,sequence_length):
    sequences = []
    labels = []

    for unit in df['unit_number'].unique():
        unit_df = df[df['unit_number'] == unit]

        for i in range(len(unit_df) - sequence_length):
            seq = unit_df[feature_cols].iloc[i:i+sequence_length].values
            label = unit_df['risk_label'].iloc[i+sequence_length]

            sequences.append(seq)
            labels.append(label)

    return np.array(sequences), np.array(labels)
