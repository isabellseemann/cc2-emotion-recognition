#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#python src/train.py experiment=cmu_glove.yaml model.model.embedding_creator.num_filters=20 model.model.classifier.input_size=60
#python src/train.py experiment=cmu_video.yaml model.model.embedding_creator.lstm_hidden_size=128 model.model.classifier.input_size=128 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=128
python src/train.py experiment=cmu_audio.yaml model.model.embedding_creator.lstm_hidden_size=64 model.model.classifier.input_size=64 model.model.embedding_creator.lstm_layers=1 model.model.embedding_creator.cnn_out_channels=64

#python src/train.py experiment=cmu_trimodal.yaml model.model.classifier.hidden_size=64 model.model.classifier.dropout_value=0.5
#
#python src/train.py experiment=cmu_trimodal.yaml skip_train_text_model=true skip_train_audio_model=true skip_train_video_model=true
#python src/train.py experiment=cmu_trimodal.yaml skip_train_text_model=false skip_train_audio_model=false skip_train_video_model=false
#python src/train.py experiment=cmu_trimodal.yaml skip_train_text_model=true skip_train_audio_model=true skip_train_video_model=false
#python src/train.py experiment=cmu_trimodal.yaml skip_train_text_model=false skip_train_audio_model=true skip_train_video_model=true
#python src/train.py experiment=cmu_trimodal.yaml skip_train_text_model=true skip_train_audio_model=false skip_train_video_model=true


