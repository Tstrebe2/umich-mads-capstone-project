import argparse

def get_argparser():
    parser = argparse.ArgumentParser(
                        prog = 'Densenet trainer.',
                        description = 'This script trains a densenet model on the chest x-ray 14 dataset.',
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
    
    parser.add_argument('--lr_scheduler_patience', 
                        nargs='?', 
                        default=3, 
                        help='Number of epochs of non-improvement experienced to decrease learning rate.', 
                        type=int, 
                        required=False)
    
    parser.add_argument('--lr_scheduler_factor', 
                        nargs='?', 
                        default=.7, 
                        help='Learning rate schedule decrease factor.', 
                        type=float, 
                        required=False)
    
    parser.add_argument('--lr_scheduler_min_lr', 
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
                        choices=[0, 1],
                        default=0, 
                        help='Boolean 0 or 1 value to indicate whether to freeze features during training.', 
                        type=int, 
                        required=False)

    parser.add_argument('--image_dir', 
                        nargs='?', 
                        default='assets/chest-xray-14/images/images', 
                        help='Directory to chest X-ray 14 training images.', 
                        required=False)

    parser.add_argument('--target_dir', 
                        nargs='?', 
                        default='assets/chest-xray-14', 
                        help='Directory to csv file with chest X-ray 14 target data.', 
                        required=False)

    parser.add_argument('--models_dir', 
                        nargs='?', 
                        default='models/cx14/', 
                        help='Directory to save models.', 
                        required=False)

    parser.add_argument('--fast_dev_run', 
                        nargs='?',
                        choices=[0, 1],
                        default=0, 
                        help='Flag (0 or 1) to run quick train & val debug session on 1 batch.',
                        type=int,
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