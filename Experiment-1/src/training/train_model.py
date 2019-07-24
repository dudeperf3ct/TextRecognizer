"""
Train model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from comet_ml import Experiment

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from training.util import train_model
import argparse


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save-model", type=int, default=False,
        help="whether or not model should be saved")
    parser.add_argument("-w", "--weights", type=str, default=True,
        help="whether or not weights should be saved)
    parser.add_argument("-m", '--model', type=str, default=Character_Model,
        help="which model to use")
    parser.add_argument("-n", '--network', type=str, default=lenet,
        help="which network architecture to use")
    parser.add_argument("-d", '--dataset', type=str, default=EMNIST,
        help="which dataset to use")
    parser.add_argument("-e", '--epochs', type=int, default=10,
        help="Number of epochs")
    parser.add_argument("-b", '--batch_size', type=int, default=32,
        help="Batch size")        
    args = vars(parser.parse_args())

    return args

def train(args, use_comet : bool = True):
    data_cls = args['dataset']
    data = data_cls()
    (x_train, y_train, x_test, y_test) = data.load_data()
    y_train = to_categorical(y_train, num_classes=y_train.shape[0])
    (x_train, y_train, x_valid, y_valid) = sklearn.train_test_split(x_train, y_train,
                                            stratify=len(y_train), test_size=0.2, random_state=42)

    print ('Training shape: ', x_train.shape, y_train.shape)
    print ('Validation shape: ', x_valid.shape, y_valid.shape)
    print ('Test shape: ', x_test.shape, y_test.shape)

    model = args['model']
    Model = model(args['network'], args['dataset']) 
    print (Model)
    
    dataset = dict({
        'x_train' : x_train,
        'y_train' : y_train,
        'x_valid' : x_valid,
        'y_valid' : y_valid,
        'x_test' : x_test,
        'y_test' : y_test
    })

    if use_comet:
        #create an experiment with your api key
        experiment = Experiment(project_name='emnist',
                                auto_param_logging=False)

    #will log metrics with the prefix 'train_'   
    with experiment.train()
        train_model(
            Model,
            dataset,
            batch_size=args['batch_size'],
            epochs=args['epochs']
            )
    
    #will log metrics with the prefix 'test_'
    with experiment.test():  
        score = model.evaluate(dataset)
        print(f'[INFO] Test evaluation: {score}')
        metrics = {
            'accuracy':score
        }
        experiment.log_metrics(metrics)    

    if args['weights']:
        Model.save_weights()

    experiment.log_parameters(args)
    experiment.log_dataset_hash(x_train) #creates and logs a hash of your data  
    

def main()
    """Run experiment."""
    args = _parse_args()
    train(args)

if __name__ == '__main__':
    main()