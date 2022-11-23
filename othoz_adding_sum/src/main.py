import torch
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
import statistics
import matplotlib.pyplot as plt
from othoz_adding_sum.src.create_dataset import generate_synthetic_dataset
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


def image_plot(loss_values, file_pathname: str, sequence_length: int):
    '''
    Image plot and save near the model path
    :param loss_values: Pandas Dataframe
    :param file_pathname: Path name to save the file
    :param sequence_length: Integer
    :return:
    '''
    loss_values[['mean_train', 'mean_test']].plot()
    plt.legend(loc='lower right')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ax = plt.gca()
    ax.set_xlim([0, loss_values.shape[0] + 1])
    ax.set_ylim([0, loss_values['mean_train'].max()])
    plt.title(f'Sequence of length {sequence_length} ')
    plt.savefig(file_pathname+'.png', dpi=100)


def train_model(training_generator, validation_generator, model,
                learning_rate: float = 0.01, epochs: int = 1000,
                tolerance: int = 10,
                clip_grad_norm: int = 100,
                log_write_folder: str = '/tmp'):

    """
    Train model with non linearity Tan h value and return the trained model and early
    :param training_generator: Pytorch Training data loader
    :param validation_generator:  Pytorch Validation data loader
    :param model:  RNN model
    :param learning_rate: Float. Default: 0.01
    :param epochs:  Epochs
    :param tolerance:  Tolerance in integer
    :param counter:  Counter in integer
    :param log_write_folder: str Log write folder
    :return: Union [Pytorch model, boolean]
    """
    # Writer
    # TODO: If needed we can pass it as a click command
    writer = SummaryWriter(log_write_folder)
    # Writing to tensorflow board
    dt_loss = datetime.now().strftime("%d_%H_%M_%S")
    # Counter value
    counter = 0
    # Instantiate loss
    criterion = torch.nn.MSELoss()
    # Instantiate optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Instantiate early_stop
    early_stop = False
    # epoch loss values
    median_train_loss_list = []
    median_test_loss_list = []
    mean_train_loss_list= []
    mean_test_loss_list = []
    for epoch in range(epochs):
        # Loop over batches in an epoch using DataLoader
        loss_train_values = []

        for val_train, target_train in training_generator:
            y_pred_train = model(val_train)  # forward step
            loss_train = criterion(y_pred_train, target_train)  # compute loss
            loss_train_values.append(loss_train.item())
            loss_train.backward()  # backprop (compute gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm) # Fixed clip
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

        #Save the raw values
        median_train_loss_list.append(median_train_loss)
        median_test_loss_list.append(median_test_loss)
        mean_train_loss_list.append(mean_train_loss)
        mean_test_loss_list.append(mean_test_loss)

        #Log the values to tensorboard
        writer.add_scalars(dt_loss, {'median_train': median_train_loss,
                           'median_test': median_test_loss,
                           'mean_train': mean_train_loss,
                           'mean_test': mean_test_loss}, epoch)
        # Printing out value every 50 epoch
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
        # Early stopping logic
        if (round(mean_test_loss - mean_train_loss, 6) < 0) and (mean_test_loss <= 0.17):
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
    loss_values = pd.DataFrame(data={'median_train': median_train_loss_list,'median_test': median_test_loss_list,
                                     'mean_train': mean_train_loss_list,'mean_test': mean_test_loss_list},
                               index=list(range(1, epoch+2)))
    return model, early_stop, loss_values


