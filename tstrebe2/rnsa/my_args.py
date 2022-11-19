import argparse

def get_argparser():
    parser = argparse.ArgumentParser(
                        prog = 'Densenet trainer.',
                        description = 'This script trains a densenet model on the RNSA pneumonia dataset.',
                        epilog = 'For help append train.py with --help')

    parser.add_argument('--batch_size', 
                        nargs='?', 
                        default=24, 
                        help='Batch size for train and val.', 
                        type=int, 
                        required=False)

    parser.add_argument('--epochs', 
                        nargs='?', 
                        default=25, 
                        help='Total number of training epochs.', 
                        type=int, 
                        required=False)

    parser.add_argument('--init_learning_rate', 
                        nargs='?', 
                        default=1e-4, 
                        help='The initial learning rate.', 
                        type=float, 
                        required=False)

    parser.add_argument('--momentum', 
                        nargs='?', 
                        default=.9, 
                        help='Optimizer momentum.', 
                        type=float, 
                        required=False)

    parser.add_argument('--weight_decay', 
                        nargs='?', 
                        default=1e-4, 
                        help='Enter weight decay.', 
                        type=float, 
                        required=False)
    
    parser.add_argument('--eta_min', 
                        nargs='?', 
                        default=1e-5, 
                        help='Minimum learning rate limit to for scheduler to decrease to.', 
                        type=float, 
                        required=False)

    parser.add_argument('--num_stop_rounds', 
                        nargs='?', 
                        default=4, 
                        help='Enter number of rounds for no improvement to trigger early stopping.', 
                        type=int, 
                        required=False)

    parser.add_argument('--freeze_features', 
                        nargs='?',
                        choices=['All', 'First3', 'None'],
                        default='False', 
                        help='Indicate whether to freeze features during training. First3 Option freezes first 3 layers of features.',
                        required=False)

    parser.add_argument('--image_dir', 
                        nargs='?', 
                        default='assets/rnsa-pneumonia/train-images', 
                        help='Directory to chest X-ray 14 training images.', 
                        required=False)

    parser.add_argument('--target_dir', 
                        nargs='?', 
                        default='assets/rnsa-pneumonia/', 
                        help='Directory to csv file with RNSA pneumonia target data.', 
                        required=False)

    parser.add_argument('--models_dir', 
                        nargs='?', 
                        default='models/rnsa/', 
                        help='Directory to save models.', 
                        required=False)
    
    parser.add_argument('--transfer_model_path', 
                        nargs='?', 
                        default='None', 
                        help='Path to model that will be used to transfer weights (feature layer) of densenet.', 
                        required=False)

    parser.add_argument('--fast_dev_run', 
                        nargs='?',
                        choices=['False', 'True'],
                        default='False', 
                        help='Flag (0 or 1) to run quick train & val debug session on 1 batch.',
                        required=False)

    parser.add_argument('--num_nodes', 
                        nargs='?', 
                        default=2, 
                        help='Number of nodes for training.',
                        type=int,
                        required=False)

    parser.add_argument('--num_workers_per_node', 
                        nargs='?', 
                        default=4, 
                        help='Number of workers from each nodeto assign to the data loader.',
                        type=int,
                        required=False)

    parser.add_argument('--restore_ckpt_path', 
                        nargs='?', 
                        default='None', 
                        help='Path to checkpoint to resume training.', 
                        required=False)
    
    return parser