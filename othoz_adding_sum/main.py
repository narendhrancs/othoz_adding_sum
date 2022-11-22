import torch
import click
import sys
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
import statistics
from typing import TypeVar, Union
from othoz_adding_sum.model_rnn_tanh import RNNModel
from othoz_adding_sum.create_dataset import generate_synthetic_dataset
logging.basicConfig(level=logging.INFO)


def get_batch_data(num_of_train_dataset:int = 100000, num_of_test_dataset:int = 10000,
                   sequence_length:int = 150, batch_size:int = 1000, shuffle:bool = False
                   ):
    """
    Divide the data to batch data
    :param num_of_train_dataset: Type: Integer. Number of training dataset
    :param num_of_test_dataset: Type: Integer. Number of test dataset
    :param sequence_length: Type: Integer. Length of the sequence
    :param batch_size: Type: Integer. Training data to be divided
    :param shuffle: If the data needs to be schuffled or not
    :return: training_dataset,validation dataset
    """
    feature_train, target_train = generate_synthetic_dataset(number_of_samples=num_of_train_dataset,
                                                             sequence_length=sequence_length)

    feature_test, target_test = generate_synthetic_dataset(number_of_samples=num_of_test_dataset,
                                                           sequence_length=sequence_length)
    # Generators
    training_set = TensorDataset(feature_train,
                                 target_train)
    training_generator = DataLoader(training_set,
                                    batch_size=batch_size,
                                    shuffle=shuffle)

    validation_set = TensorDataset(feature_test,
                                   target_test)
    validation_generator = DataLoader(validation_set,
                                      batch_size=batch_size,
                                      shuffle=shuffle)
    return training_generator, validation_generator


def train_model(training_generator, validation_generator, model,
                learning_rate: float = 0.01, epochs:int = 1000,
                tolerance:int = 10):

    """
    Train model with non linearity Tan h value and return the trained model and early
    :param training_generator: Pytorch Training data loader
    :param validation_generator:  Pytorch Validation data loader
    :param model:  RNN model
    :param learning_rate: Float. Default: 0.01
    :param epochs:  Epochs
    :param tolerance:  Tolerance in integer
    :param counter:  Counter in integer
    :return: Union [Pytorch model, boolean]
    """
    # Writer
    # TODO: If needed we can pass it as a click command
    writer = SummaryWriter('/tmp')
    # Writing to tensorflow board
    dt_loss = datetime.now().strftime("%d_%H_%M_%S")
    # Counter value
    counter = 0
    # Instantiate loss
    criterion = torch.nn.MSELoss()
    # Instantiate optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # Loop over batches in an epoch using DataLoader
        loss_train_values = []

        for val_train, target_train in training_generator:
            y_pred_train = model(val_train)  # forward step
            loss_train = criterion(y_pred_train, target_train)  # compute loss
            loss_train_values.append(loss_train.item())
            loss_train.backward()  # backprop (compute gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100) # Fixed clip
            optimizer.step()  # update weights (gradient descent step)
            optimizer.zero_grad()  # reset gradients

        with torch.no_grad():
            loss_test_values = []
            for val_test, target_test in validation_generator:
                y_pred_test = model(val_test)  # forward step
                loss_test = criterion(y_pred_test, target_test)  # compute loss
                loss_test_values.append(loss_test.item())

        median_train_loss = round(statistics.median(loss_train_values), 6)
        median_test_loss = round(statistics.median(loss_test_values), 6)

        mean_train_loss = round(statistics.mean(loss_train_values), 6)
        mean_test_loss = round(statistics.median(loss_test_values), 6)

        # Printing out value every 50 epoch
        writer.add_scalars(dt_loss, {'median_train': median_train_loss,'median_test': median_test_loss,
                                     'mean_train': mean_train_loss,':mean_test': mean_test_loss},epoch)
        if (epoch % 50) == 0:
            logging.info(f"[EPOCH]: {epoch}, "
                         f"[Median LOSS TRAIN]:{median_train_loss:.6f}"
                         f"[Median LOSS TEST]: {median_test_loss:.6f}, "
                         )

            mean_test_loss = round(statistics.median(loss_test_values), 6)
            mean_train_loss = round(statistics.mean(loss_train_values), 6)
            logging.info(f"[EPOCH]: {epoch}, "
                         f"[Mean LOSS TRAIN]:{mean_train_loss:.6f}"
                         f"[Mean LOSS TEST]: {mean_test_loss:.6f},"
                         )

        if (round(mean_test_loss - mean_train_loss, 6) > 0) and (mean_test_loss <= 0.17):
            counter += 1
            if counter >= tolerance:
                logging.info(f"[EPOCH]: {epoch}, "
                             f"[Mean LOSS TRAIN]:{mean_train_loss:.6f}"
                             f"[Mean LOSS TEST]: {mean_test_loss:.6f},"
                             )
                early_stop = True
                logging.info('EARLY STOP')
                break
    writer.close()
    return model, early_stop

@click.command()
@click.option('--modelpath', default='/tmp/add_sum', help='Place to save the model path. Default path is /tmp/add_sum')
@click.option('--num_of_train_dataset', default=100000, help='Num of training samples to use for training. Default is 100000')
@click.option('--num_of_test_dataset', default=10000, help='Num of test samples to use for testing. Default is 10000')
@click.option('--batch_size', default=100, help='Batch size to use for training the model. Default is 100')
@click.option('--sequence_length', default=150, help='sequence length to use for the model. Default is 150')
@click.option('--rnn_nonlinearity', default='tanh', help='Non linearity of the RNN model. Default is tanh')
def main(modelpath:str = "/tmp/add_sum",
         num_of_train_dataset:int = 10000,
         num_of_test_dataset:int = 1000,
         batch_size:int = 100,
         shuffle:bool = False,
         sequence_length:int = 150,
         rnn_nonlinearity:str = 'tanh'
         ) -> None:
    """
    The main function to train and save the model.

    :param filepath: str Path of the saved model or model to be saved
    :param num_of_train_dataset: int
    :param num_of_test_dataset: int
    :param batch_size: int
    :param shuffle: bool
    :param sequence_length: int
    :param nonlinearity: str
    :return: None Saved model
    """
    # Get batch data
    training_generator, validation_generator = get_batch_data(num_of_train_dataset, num_of_test_dataset,
                                                              sequence_length, batch_size, shuffle)
    # Get the data set size for printing (it is equal to N_SAMPLES)
    logging.info(f'Training data size {len(training_generator.dataset)}')

    # Initialise RNN MODEL
    rnn_tanh_model = RNNModel(nonlinearity=rnn_nonlinearity)
    # Print model & Parameters
    logging.info(f'RNN TANH model: {rnn_tanh_model}')
    #
    model, early_stop = train_model(training_generator, validation_generator, rnn_tanh_model,
                                    learning_rate=0.01, epochs=1000)
    if early_stop:
        # Save the entire model
        torch.save(model,
               modelpath+rnn_nonlinearity+str(sequence_length)+'.p')
    else:
        logging.critical('No early stop was found')
        sys.exit(1)

if __name__ == "__main__":
    main()