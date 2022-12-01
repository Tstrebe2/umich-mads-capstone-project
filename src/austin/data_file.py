import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

cx14_path = 'data/chest-xray-14/Data_Entry_2017_v2020.csv'
rsna_path = 'data/rsna-pneumonia/stage_2_train_labels.csv'


def add_split(df):
    train_df, val_df = train_test_split(df, train_size=0.8, random_state=42, shuffle=True, stratify=df['Target'])
    val_df, test_df = train_test_split(val_df, train_size=0.5, random_state=42, shuffle=True, stratify=val_df['Target'])
    train_ix, val_ix, test_ix = list(train_df.index), list(val_df.index), list(test_df.index)
    df['split'] = df.apply(lambda x: 'train' if x.name in train_ix else 'val' if x.name in val_ix else 'test' if x.name in test_ix else 'none',
    axis = 1)
    return df


rsna_df = add_split(pd.read_csv(rsna_path))
rsna_df.to_csv('rsna_targets.csv', index=True, index_label='index')

diseases = ['Pneumonia','Infiltration','Consolidation','Atelectasis']
def pneumo_like(cell):
    conditions = cell.split('|')
    for condition in conditions:
        if condition in diseases:
            return 1
    return 0

cx14_df = pd.read_csv(cx14_path)
cx14_df['Target'] = cx14_df['Finding Labels'].apply(pneumo_like)
cx14_df = add_split(cx14_df)
cx14_df.to_csv('cx14_targets.csv', index=True, index_label='index')