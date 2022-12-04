import argparse
import torch
import pandas as pd
import models
import data
import os
import gc

def print_batch_count(curr_batch:int, data_loader) -> None:
    print('{:,} of {:,} batches completed.'.format(curr_batch, len(data_loader)))

def main() -> None:
    parser = argparse.ArgumentParser(
                        prog = 'Densenet trainer.',
                        description = 'This script evals and stores results of densenet model on the RNSA pneumonia dataset.',
                        epilog = 'tstrebe2@gmail.com is my email')
    parser.add_argument('--model_dir', 
                        nargs='?', 
                        default='/home/tstrebel/repos/umich-mads-capstone-project/models/', 
                        help='Directory to model.', 
                        required=False)
    parser.add_argument('--model_name', 
                        nargs='?', 
                        help='File name of model.', 
                        required=True)
    parser.add_argument('--batch_size', 
                        nargs='?', 
                        default=32, 
                        help='Batch size for train and val.', 
                        type=int, 
                        required=False)
    parser.add_argument('--img_dir', 
                        nargs='?', 
                        default='/home/tstrebel/assets/chest-xray-14/images/images/', 
                        help='Directory to where images are stored.', 
                        required=False)
    parser.add_argument('--patient_data_path', 
                        nargs='?', 
                        default='/home/tstrebel/repos/umich-mads-capstone-project/assets/cx14-inference.csv', 
                        help='Path to patient load/save patient detailed information where predictions will be stored.', 
                        required=False)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv('/home/tstrebel/repos/umich-mads-capstone-project/assets/cx14-targets.csv')#args.patient_data_path)
    
    dataset = data.get_dataset(args.img_dir, df)
    data_loader = data.get_data_loader(dataset, batch_size=args.batch_size)

    model_dict = torch.load(os.path.join(args.model_dir, args.model_name), map_location=device)

    model = models.Densenet121()
    model.load_state_dict(model_dict['state_dict'])

    model = model.to(device)
    print('Model loaded successfully')
    print('begining inferencing')
    
    curr_batch = 0
    print_batch_count(curr_batch, data_loader)
    with torch.no_grad():
        running_targets = torch.Tensor(0, 1).to(device)
        running_outputs = torch.Tensor(0, 1).to(device)

        model.eval()
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            running_targets = torch.vstack((running_targets, targets))
            running_outputs = torch.vstack((running_outputs, outputs))
            
            curr_batch += 1
            print_batch_count(curr_batch, data_loader)
    
    ix = len(args.model_name) if '.' not in args.model_name else args.model_name.index('.')
    
    col_name = args.model_name[:ix].replace('-', '_')
    col_name = f'{col_name}_proba'
    
    if col_name in df.columns:
        df.drop(col_name, axis=1, inplace=True)

    df[col_name] = torch.sigmoid(running_outputs.cpu()).numpy()
    
    df.to_csv(args.patient_data_path, index=False)
    
if __name__ == '__main__':
    main()
