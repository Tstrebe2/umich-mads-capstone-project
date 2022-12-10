import pandas as pd
import numpy as np
import torch
import models
import data
import argparse

def main():
    parser = argparse.ArgumentParser(
                        prog = 'Densenet image feature extractor.',
                        description = 'This script obtains image feature from the hidden layers of a pretrained Densenet model \
                            by bypassing the classifier on the forward pass.',
                        epilog = 'tstrebe2@gmail.com is my email')
    
    parser.add_argument('--model_path', 
                        help='File path to densenet model.', 
                        required=True)
    
    parser.add_argument('--save_path', 
                        help='File path to save final model to.', 
                        required=True)

    parser.add_argument('--img_dir', 
                        nargs='?', 
                        default='/home/tstrebel/assets/rsna-pneumonia/train-images/', 
                        help='Directory where images are stored.', 
                        required=False)

    parser.add_argument('--targets_path', 
                        nargs='?', 
                        default='/home/tstrebel/repos/umich-mads-capstone-project/assets/rsna-targets.csv', 
                        help='File path to get target data from.', 
                        required=False)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(args.model_path, map_location=device)['state_dict']

    model = models.Densenet121FeatureExtractor()

    _ = model.load_state_dict(state_dict)

    model = model.to(device)

    print('Model loaded')

    data_dict = data.get_training_data_target_dict(args.targets_path, False)
    df = pd.concat([v for v in data_dict.values()])

    dataset = data.get_dataset(args.img_dir, df, indexed=True)
    data_loader = data.get_data_loader(dataset, batch_size=32)

    results = np.empty_like((), shape=(0, 1028))

    total_batches = len(data_loader)

    print('Starting process of extracting features from {:,} batches of images.'.format(total_batches))

    counter = 0

    with torch.no_grad():
        for indices, inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            indices = indices.cpu().numpy()

            outputs = model(inputs)

            features = outputs.squeeze().cpu().numpy()
            targets = targets.cpu().numpy()
            patient_ids = df.iloc[indices][['patient_id']].values
            splits = df.iloc[indices][['split']].values

            batch = np.hstack((indices.reshape(-1, 1), patient_ids, features, targets, splits))
            results = np.vstack((results, batch))

            print('{:,} of {:,} image features extrated'.format(counter + 1, total_batches))
            counter += 1

    df_final = pd.DataFrame(results, columns=['index', 'patient_id'] + ['f{}'.format(n) for n in range(1024)] + ['target', 'split']).set_index('index')
    df_final.to_csv(args.save_path, index_label='index')
    
if __name__ == '__main__':
    main()