#/bin/sh
echo "Run the tensorboard in background"
tensorboard --logdir='/tmp' --host localhost --port 6066 &

echo "Train the model"
python tutorial/train_model.py --modelpath='/tmp/' --num_of_train_dataset=100 --num_of_test_dataset=100 \
--batch_size=100 --sequence_length=150 --rnn_nonlinearity='tanh' --clip_grad_norm=100 \
--log_write_folder='/tmp/' --learning_rate=0.01 --epochs=100 --shuffle=False --model_type="rnn"

echo "Inference of the model"
python tutorial/inference.py --model_filepath='/tmp/150.p'