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
from src.data.emnist_lines import EMNISTLines
from src.data.iam_lines import IAMLines
from src.models.line_model import LineModel
from src.models.line_model_ctc import LineModelCTC
from src.networks.lenet_cnn import lenetcnn, lenetcnnslide
from src.networks.resnet_cnn import resnetcnn
from src.networks.custom_cnn import customcnn
from src.networks.cnn_lstm_ctc import cnnlstmctc

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
    parser.add_argument("-m", '--model', type=str, default='LineModel',
        help="which model to use")
    parser.add_argument("-n", '--network', type=str, default='lenetcnn',
        help="which network architecture to use")
    parser.add_argument("-d", '--dataset', type=str, default='EMNISTLines',
        help="which dataset to use")
    parser.add_argument("-bb", '--backbone', type=str, default='lenet',
        help="which backbone architecture to use (only for lstm ctc network)")
    parser.add_argument("-sq", '--seq', type=str, default="lstm",
        help="which sequence model to use (only for lstm ctc network)")
    parser.add_argument("-bi", '--bi', type=bool, default=False,
        help="whether to use bidirectional model (only for lstm ctc network)")
    parser.add_argument("-e", '--epochs', type=int, default=10,
        help="Number of epochs")
    parser.add_argument("-b", '--batch_size', type=int, default=32,
        help="Batch size") 
    parser.add_argument("-find_lr", '--find_lr', type=bool, default=False,
        help="Find lr")        
    args = vars(parser.parse_args())

    return args


funcs = {'EMNISTLines': EMNISTLines, 'IAMLines': IAMLines, 'lenetcnn': lenetcnn, 'resnetcnn': resnetcnn,
        'customcnn': customcnn, 'LineModel': LineModel, 'lenetcnnslide': lenetcnnslide,
        'lstmctc': cnnlstmctc, 'LineModelCTC': LineModelCTC}

def train(args, use_comet : bool = True):

    data_cls = funcs[args['dataset']]
    model_cls = funcs[args['model']]
    network = funcs[args['network']]

    print ('[INFO] Getting dataset...')
    data = data_cls()
    data.load_data()
    (x_train, y_train), (x_test, y_test) = (data.x_train, data.y_train), (data.x_test, data.y_test)
    classes = data.mapping
    
    # #Used for testing only
    # x_train = x_train[:100, :, :]
    # y_train = y_train[:100, :]
    # x_test = x_test[:100, :, :]
    # y_test = y_test[:100, :]
    # print ('[INFO] Training shape: ', x_train.shape, y_train.shape)
    # print ('[INFO] Test shape: ', x_test.shape, y_test.shape)
    # #delete these lines

    # distribute 90% test 10% val dataset with equal class distribution 
    (x_test, x_valid, y_test, y_valid) = train_test_split(x_test, y_test, test_size=0.2, random_state=42)

    print ('[INFO] Training shape: ', x_train.shape, y_train.shape)
    print ('[INFO] Validation shape: ', x_valid.shape, y_valid.shape)
    print ('[INFO] Test shape: ', x_test.shape, y_test.shape)

    print ('[INFO] Setting up the model..')
    if args['network'] == 'lstmctc':
        network_args = {'backbone' : args['backbone'],
                        'seq_model' : args['seq'],
                        'bi' : args['bi']
                        }
        model = model_cls(network, data_cls, network_args)
    else:
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
        experiment = Experiment(api_key='WVBNRAfMLCBWslJAAsffxM4Gz',
                                project_name='iam_lines',
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
            score = model.evaluate(dataset, int(args['batch_size']))
            print(f'[INFO] Test evaluation: {score*100}...')
            metrics = {
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
        score = model.evaluate(dataset, args['batch_size'])
        print(f'[INFO] Test evaluation: {score*100}...')

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