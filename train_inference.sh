#/bin/sh
set -e

echo "Run the tensorboard in background"
tensorboard --logdir='/tmp' --host '0.0.0.0' --port 6006 &


echo "Train the model with rnn sequence_length 150"
python tutorial/train_model.py --modelpath='/tmp/' --num_of_train_dataset=10000 --num_of_test_dataset=1000 \
--batch_size=100 --sequence_length=150 --rnn_nonlinearity='tanh' --clip_grad_norm=100 \
--log_write_folder='/tmp/' --learning_rate=0.01 --epochs=1000 --shuffle=False

echo "Train the model with rnn sequence_length 200"
python tutorial/train_model.py --modelpath='/tmp/' --num_of_train_dataset=10000 --num_of_test_dataset=1000 \
--batch_size=100 --sequence_length=200 --rnn_nonlinearity='tanh' --clip_grad_norm=100 \
--log_write_folder='/tmp/' --learning_rate=0.01 --epochs=1000 --shuffle=False

echo "Train the model with rnn sequence_length 300"
python tutorial/train_model.py --modelpath='/tmp/' --num_of_train_dataset=10000 --num_of_test_dataset=1000 \
--batch_size=100 --sequence_length=300 --rnn_nonlinearity='tanh' --clip_grad_norm=100 \
--log_write_folder='/tmp/' --learning_rate=0.01 --epochs=1000 --shuffle=False

echo "Train the model with rnn sequence_length 400"
python tutorial/train_model.py --modelpath='/tmp/' --num_of_train_dataset=10000 --num_of_test_dataset=1000 \
--batch_size=100 --sequence_length=400 --rnn_nonlinearity='tanh' --clip_grad_norm=100 \
--log_write_folder='/tmp/' --learning_rate=0.01 --epochs=1000 --shuffle=False