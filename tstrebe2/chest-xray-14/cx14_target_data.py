import os
import pandas as pd
from sklearn.model_selection import train_test_split

def get_training_data_target_dict(target_dir:str) -> dict:
    # Define our target variables
    target_map = {'No Finding':0, 'Atelectasis':1, 'Cardiomegaly':2, 'Consolidation':3, 'Edema':4,
           'Effusion':5, 'Emphysema':6, 'Fibrosis':7, 'Hernia':8, 'Infiltration':9,
           'Mass':10, 'Nodule':11, 'Pleural_Thickening':12, 'Pneumonia':13, 'Pneumothorax':14}
    
    target_df = pd.read_csv(os.path.join(target_dir, 'Data_Entry_2017.csv'), usecols=[0, 1])
    target_df.columns = ['file_path', 'target']
    
    # We're only using labels defined by the train_val dataset
    train_val_df = pd.read_csv(os.path.join(target_dir, 'train_val_list.txt'), header=None, index_col=0).index
    target_df = target_df[target_df['file_path'].isin(train_val_df)]
        
    # Remove multi-condition labels from out dataset
    target_df = target_df[~(target_df.target.str.contains('\|'))]
    # Map targets strings to integers
    target_df.target = target_df.target.map(target_map)

    #Set random_state to 99 for reproduceability
    X_train, X_val = train_test_split(target_df, stratify=target_df.target, test_size=.2, random_state=99)
    X_val, X_test = train_test_split(X_val, stratify=X_val.target, test_size=.4, random_state=99)
    train_ix, val_ix, test_ix = list(X_train.index), list(X_val.index), list(X_test.index)
    
    data_dict = {
        'target_map':target_map,
        'df_train':target_df.loc[train_ix],
        'df_val':target_df.loc[val_ix],
        'df_test':target_df.loc[test_ix],
    }
    
    return data_dict