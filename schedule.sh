#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py experiment=cmu_glove.yaml model.model.embedding_creator.num_filters=50 model.model.classifier.input_size=150
python src/train.py experiment=cmu_glove.yaml model.model.embedding_creator.num_filters=100 model.model.classifier.input_size=300
python src/train.py experiment=cmu_glove.yaml model.model.embedding_creator.num_filters=20 model.model.classifier.input_size=60

python src/train.py experiment=cmu_audio.yaml model.model.embedding_creator.lstm_hidden_size=64 model.model.classifier.input_size=64 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=64
python src/train.py experiment=cmu_audio.yaml model.model.embedding_creator.lstm_hidden_size=64 model.model.classifier.input_size=64 model.model.embedding_creator.lstm_layers=2 model.model.embedding_creator.cnn_out_channels=64
python src/train.py experiment=cmu_audio.yaml model.model.embedding_creator.lstm_hidden_size=64 model.model.classifier.input_size=64 model.model.embedding_creator.lstm_layers=3 model.model.embedding_creator.cnn_out_channels=64
python src/train.py experiment=cmu_audio.yaml model.model.embedding_creator.lstm_hidden_size=32 model.model.classifier.input_size=32 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=32
python src/train.py experiment=cmu_audio.yaml model.model.embedding_creator.lstm_hidden_size=32 model.model.classifier.input_size=32 model.model.embedding_creator.lstm_layers=2 model.model.embedding_creator.cnn_out_channels=32
python src/train.py experiment=cmu_audio.yaml model.model.embedding_creator.lstm_hidden_size=32 model.model.classifier.input_size=32 model.model.embedding_creator.lstm_layers=3 model.model.embedding_creator.cnn_out_channels=32
python src/train.py experiment=cmu_audio.yaml model.model.embedding_creator.lstm_hidden_size=128 model.model.classifier.input_size=128 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=128
python src/train.py experiment=cmu_audio.yaml model.model.embedding_creator.lstm_hidden_size=256 model.model.classifier.input_size=256 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=256
python src/train.py experiment=cmu_audio.yaml model.model.embedding_creator.lstm_hidden_size=64 model.model.classifier.input_size=64 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=256
python src/train.py experiment=cmu_audio.yaml model.model.embedding_creator.lstm_hidden_size=256 model.model.classifier.input_size=256 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=64
python src/train.py experiment=cmu_audio.yaml model.model.embedding_creator.lstm_hidden_size=64 model.model.classifier.input_size=64 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=256
python src/train.py experiment=cmu_audio.yaml model.model.embedding_creator.lstm_hidden_size=256 model.model.classifier.input_size=256 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=64

python src/train.py experiment=cmu_video.yaml model.model.embedding_creator.lstm_hidden_size=64 model.model.classifier.input_size=64 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=64
python src/train.py experiment=cmu_video.yaml model.model.embedding_creator.lstm_hidden_size=64 model.model.classifier.input_size=64 model.model.embedding_creator.lstm_layers=2 model.model.embedding_creator.cnn_out_channels=64
python src/train.py experiment=cmu_video.yaml model.model.embedding_creator.lstm_hidden_size=64 model.model.classifier.input_size=64 model.model.embedding_creator.lstm_layers=3 model.model.embedding_creator.cnn_out_channels=64
python src/train.py experiment=cmu_video.yaml model.model.embedding_creator.lstm_hidden_size=64 model.model.classifier.input_size=32 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=32
python src/train.py experiment=cmu_video.yaml model.model.embedding_creator.lstm_hidden_size=64 model.model.classifier.input_size=32 model.model.embedding_creator.lstm_layers=2 model.model.embedding_creator.cnn_out_channels=32
python src/train.py experiment=cmu_video.yaml model.model.embedding_creator.lstm_hidden_size=64 model.model.classifier.input_size=32 model.model.embedding_creator.lstm_layers=3 model.model.embedding_creator.cnn_out_channels=32
python src/train.py experiment=cmu_video.yaml model.model.embedding_creator.lstm_hidden_size=128 model.model.classifier.input_size=128 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=128
python src/train.py experiment=cmu_video.yaml model.model.embedding_creator.lstm_hidden_size=256 model.model.classifier.input_size=256 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=256
python src/train.py experiment=cmu_video.yaml model.model.embedding_creator.lstm_hidden_size=64 model.model.classifier.input_size=64 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=256
python src/train.py experiment=cmu_video.yaml model.model.embedding_creator.lstm_hidden_size=256 model.model.classifier.input_size=256 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=64

python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=128
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=128 model.model.classifier.dropout_value=0.3
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=128 model.model.classifier.dropout_value=0.5
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=128 model.model.classifier.dropout_value=0.7
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=256
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=256 model.model.classifier.dropout_value=0.3
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=256 model.model.classifier.dropout_value=0.5
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=256 model.model.classifier.dropout_value=0.7
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=64
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=64 model.model.classifier.dropout_value=0.3
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=64 model.model.classifier.dropout_value=0.5
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=64 model.model.classifier.dropout_value=0.7
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=32
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=32 model.model.classifier.dropout_value=0.3
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=32 model.model.classifier.dropout_value=0.5
python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=32 model.model.classifier.dropout_value=0.7



