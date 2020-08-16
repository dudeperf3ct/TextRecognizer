"""
Train model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from comet_ml import Experiment

from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from sklearn.model_selection import train_test_split
from src.training.util import train_model
from src.data.emnist_dataset import EMNIST
from src.models.character_model import Character_Model
from src.networks.lenet import lenet
from src.networks.resnet import resnet
from src.networks.custom import customCNN
import argparse

def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save-model", type=int, default=False,
        help="whether or not model should be saved")
    parser.add_argument("-w", "--weights", type=str, default=True,
        help="whether or not weights should be saved")
    parser.add_argument("-save_model", "--save_model", type=str, default=False,
        help="whether or not model should be saved")
    parser.add_argument("-m", '--model', type=str, default="Character_Model",
        help="which model to use")
    parser.add_argument("-n", '--network', type=str, default="lenet",
        help="which network architecture to use")
    parser.add_argument("-d", '--dataset', type=str, default="EMNIST",
        help="which dataset to use")
    parser.add_argument("-e", '--epochs', type=int, default=10,
        help="Number of epochs")
    parser.add_argument("-b", '--batch_size', type=int, default=32,
        help="Batch size") 
    parser.add_argument("-find_lr", '--find_lr', type=bool, default=False,
        help="Find lr")        
    args = vars(parser.parse_args())

    return args


funcs = {'EMNIST': EMNIST, 'lenet': lenet, 'resnet' : resnet, 'customCNN' : customCNN, 'Character_Model': Character_Model}


def train(args, use_comet : bool = True):

    data_cls = funcs[args['dataset']]
    model_cls = funcs[args['model']]
    network = funcs[args['network']]

    print ('[INFO] Getting dataset...')
    data = data_cls()
    (x_train, y_train), (x_test, y_test) = data.load_data()
    classes = data.mapping
    
    # #Used for testing only
    # x_train = x_train[:100, :, :]
    # y_train = y_train[:100, :]
    # x_test = x_test[:100, :, :]
    # y_test = y_test[:100, :]
    # print ('[INFO] Training shape: ', x_train.shape, y_train.shape)
    # print ('[INFO] Test shape: ', x_test.shape, y_test.shape)
    # #delete these lines

    y_test_labels = [np.where(y_test[idx]==1)[0][0] for idx in range(len(y_test))]
    # distribute 90% test 10% val dataset with equal class distribution 
    (x_test, x_valid, y_test, y_valid) = train_test_split(x_test, y_test, test_size=0.1,
                                            stratify=y_test_labels, random_state=42)

    print ('[INFO] Training shape: ', x_train.shape, y_train.shape)
    print ('[INFO] Validation shape: ', x_valid.shape, y_valid.shape)
    print ('[INFO] Test shape: ', x_test.shape, y_test.shape)

    print ('[INFO] Setting up the model..')
    model = model_cls(network, data_cls)
    print (model)
    
    dataset = dict({
        'x_train' : x_train,
        'y_train' : y_train,
        'x_valid' : x_valid,
        'y_valid' : y_valid,
        'x_test' : x_test,
        'y_test' : y_test
    })

    if use_comet and args['find_lr'] == False:
        #create an experiment with your api key
        experiment = Experiment(api_key='INSERT API KEY',
                                project_name='emnist',
                                auto_param_logging=False)
        
        print ('[INFO] Starting Training...')
        #will log metrics with the prefix 'train_'   
        with experiment.train():
            _ = train_model(
                    model,
                    dataset,
                    batch_size=args['batch_size'],
                    epochs=args['epochs'],
                    name=args['network']
                    )

        print ('[INFO] Starting Testing...')    
        #will log metrics with the prefix 'test_'
        with experiment.test():  
            loss, score = model.evaluate(dataset, args['batch_size'])
            print(f'[INFO] Test evaluation: {score*100}')
            metrics = {
                'loss':loss,
                'accuracy':score
            }
            experiment.log_metrics(metrics)    

        experiment.log_parameters(args)
        experiment.log_dataset_hash(x_train) #creates and logs a hash of your data 
        experiment.end()

    elif use_comet and args['find_lr'] == True:

        _ = train_model(
                    model,
                    dataset,
                    batch_size=args['batch_size'],
                    epochs=args['epochs'],
                    FIND_LR=args['find_lr'],
                    name=args['network']
                    )

    else :

        print ('[INFO] Starting Training...')
        train_model(
            model,
            dataset,
            batch_size=args['batch_size'],
            epochs=args['epochs'],
            name=args['network']
            )
        print ('[INFO] Starting Testing...')    
        loss, score = model.evaluate(dataset, args['batch_size'])
        print(f'[INFO] Test evaluation: {score*100}')

    if args['weights']:
        model.save_weights()
    
    if args['save_model']:
        model.save_model()
        

def main():
    """Run experiment."""
    args = _parse_args()
    train(args)

if __name__ == '__main__':
    main()
