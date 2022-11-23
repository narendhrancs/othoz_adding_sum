import torch
import numpy as np
import logging
import sys
from othoz_adding_sum.src.custom_exception import EXCEPTION_NO_OF_SAMPLES_LESS, EXCEPTION_SEQUENCE_LENGTH_LESS


def generate_synthetic_dataset(number_of_samples:int, sequence_length:int, uniform_range:tuple = (0,1)):
    """
    Generate synthetic data for the adding problem.
    Each row in a sequence_length contains Random signal as uniform value  and Mask with binary value.
    Dataset format: (number_of_samples, sequence_length, 2)
    :param number_of_samples: integer
    :param sequence_length: integer
    :param uniform_range:tuple
    :return: torch value and target value in vector format
    """
    try:
        if number_of_samples < 1:
            logging.error('number_of_samples should be more than 1')
            raise EXCEPTION_NO_OF_SAMPLES_LESS
        elif sequence_length < 2:
            logging.error('sequence_length should be more than 2')
            raise EXCEPTION_SEQUENCE_LENGTH_LESS


        # Set seed value for randomness
        torch.manual_seed(42)
        np.random.seed(0)
        # Uniform distribution
        rand_a = torch.distributions.uniform\
                                    .Uniform(uniform_range[0], uniform_range[1])\
                                    .sample([number_of_samples, sequence_length, 1])
        # Initialise
        # Set mask_a values as zeros
        mask_a = torch.zeros(number_of_samples, sequence_length, 1)
        # Initialise
        # Set target values as zeros. Initialise
        target = torch.zeros(number_of_samples, 1)

        # Set the two mask value and the sum of two target values
        for sample in range(number_of_samples):
            positions = np.random.choice(sequence_length, size=2, replace=False)
            mask_a[sample, positions[0]] = 1
            mask_a[sample, positions[1]] = 1
            target[sample] = rand_a[sample, positions[0]]+rand_a[sample, positions[1]]

        # Concat random and mask value together
        value = torch.cat((rand_a, mask_a), 2)
        return value, target
    except (EXCEPTION_NO_OF_SAMPLES_LESS, EXCEPTION_SEQUENCE_LENGTH_LESS):
        sys.exit(1)
    except Exception as e:
        logging.critical(f'Unknown error: {e}')
        raise e