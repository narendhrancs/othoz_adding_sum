import logging
import torch
import sys
from othoz_adding_sum.src.main import get_batch_data, train_model, image_plot
from othoz_adding_sum.src.model_rnn_tanh import RNNModel
import click
logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--modelpath', default='/tmp/', help='Place to save the model path. Default path is /tmp/')
@click.option('--num_of_train_dataset', default=100000, help='Num of training samples to use for training. Default is 100000')
@click.option('--num_of_test_dataset', default=10000, help='Num of test samples to use for testing. Default is 10000')
@click.option('--batch_size', default=100, help='Batch size to use for training the model. Default is 100')
@click.option('--sequence_length', default=150, help='sequence length to use for the model. Default is 150')
@click.option('--rnn_nonlinearity', default='tanh', help='Non linearity of the RNN model.Model is tanh or relu. Default is tanh')
@click.option('--clip_grad_norm', default=100, help='Clip grad norm. Default is 100')
@click.option('--log_write_folder', default='/tmp', help='Logs to be writtenn. Default is /tmp folder')
@click.option('--learning_rate', default=0.01, help='Learning rate. Default is 0.01')
@click.option('--epochs', default=1000, help='Epochs rate. Default is 1000')
@click.option('--shuffle', default=False, help='Shuffle the data. Default is False')
def main(modelpath:str,
         num_of_train_dataset:int,
         num_of_test_dataset:int,
         batch_size:int,
         shuffle:bool,
         sequence_length:int,
         rnn_nonlinearity:str,
         log_write_folder:str,
         clip_grad_norm:int,
         learning_rate:int,
         epochs:int
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
    :param log_write_folder: str
    :param clip_grad_norm: str
    :param model_type: str Type of the model. If it is RNN or LSTM
    :return: None Saved model
    """
    # Get batch data
    training_generator, validation_generator = get_batch_data(num_of_train_dataset, num_of_test_dataset,
                                                              sequence_length, batch_size, shuffle)
    # Get the data set size for printing (it is equal to N_SAMPLES)
    logging.info(f'Training data size {len(training_generator.dataset)}')

    # Initialise RNN MODEL
    model = RNNModel(nonlinearity=rnn_nonlinearity)
    # Print model & Parameters
    logging.info(f'RNN TANH model: {model}')
    logging.info(f'Non linearity: {rnn_nonlinearity}')

    #
    model, early_stop,loss_values = train_model(training_generator,
                                    validation_generator,
                                    model,
                                    learning_rate=learning_rate, epochs=epochs,
                                    clip_grad_norm=clip_grad_norm,
                                    log_write_folder=log_write_folder
                                    )

    loss_values.to_csv(f'{log_write_folder}/loss_values_{rnn_nonlinearity}_{sequence_length}.csv',sep=',', header=True)

    # Save the epochs as a plot
    image_plot(loss_values, file_pathname=modelpath+str(sequence_length), sequence_length=sequence_length)

    if early_stop:
        # Save the entire model
        torch.save(model,
                   modelpath+str(sequence_length)+'.p')
    else:
        logging.critical('No early stop was found')
        sys.exit(1)


if __name__ == "__main__":
    main()