import click
import torch
from othoz_adding_sum.src.create_dataset import generate_synthetic_dataset
class Inference(object):
    def __init__(self, model_path:str) -> None:
        """
        Initialise the model
        :param model_path: str Model path
        """
        self.model_path=model_path
        # Load the model
        self.model = torch.load(self.model_path)
        self.model.eval()

    def forecast(self, data):
        with torch.no_grad():
            y_pred = self.model(data)  # forward step
            return y_pred
@click.command()
@click.option('--model_filepath', default='/tmp/150.p', help='Place to save the model path. Default path is /tmp/150.p')
def main(model_filepath:str):
    """
    Load the model_path and do inference.
    :param model_path: str Model path
    :return:
    """
    # Dataset for testing
    feature, target = generate_synthetic_dataset(number_of_samples=10,
                                                 sequence_length=10)
    #Load the model. Initialising early will save time in production.
    inference = Inference(model_filepath)
    # Prediction of model
    y_pred = inference.forecast(feature)
    print(y_pred)
    print(target)
    return y_pred

if __name__ == "__main__":
    main()