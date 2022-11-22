import logging
import torch
import sys
from src.main import get_batch_data, RNNModel, train_model


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

main()