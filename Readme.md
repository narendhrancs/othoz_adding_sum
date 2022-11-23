
# Purpose
Evaluation that shows how the model performs depending on
training progress (epochs) as well as input sequence length. Please also demonstrate that the
model does not exhibit overfitting.

# Push python package to PYPI
1. Execute `python setup.py dist` in terminal
2. `pip install twine`
3. `twine upload dist/*`
4. Enter the username and password 
5. Install the desired package by `pip install othoz-adding-sum`


# Manual install using setup.py in a virtual environment or using pypi
1. Manual install:  
   1.1 Login into the desired package and execute `pip install .` or editable `pip install -e .`
   1.2 `pip install --upgrade othoz_adding_sum`
2. Sample code on how to run the tutorial is mentioned in `tutorial` folder
3. During training the file provides 3 output.
    * Tensorboard output log file. 
    Open the output in browser by going to `http://localhost:6006/`
    * Model output
    * Epoch logs as a csv file

#Model 
## Training the model:
Execute the below command in terminal to get the desired output
```
python tutorial/train_model.py --modelpath='/tmp/' --num_of_train_dataset=100 --num_of_test_dataset=100 \
--batch_size=100 --sequence_length=150 --rnn_nonlinearity='tanh' --clip_grad_norm=100 \
--log_write_folder='/tmp/' --learning_rate=0.01 --epochs=100 --shuffle=False --model_type="rnn"
```

## Inference of the model:
Execute the below command in terminal to run the inferred output
```
python tutorial/inference.py --model_filepath='/tmp/150.p'
```

## Train and inference of the model.
```
sh train_inference.sh
```

# Models:
1. RNN+TANH
2. RNN+RELU
3. LSTM















































































































































































































































