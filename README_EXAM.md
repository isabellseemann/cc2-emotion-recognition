# Load the CMU-MOSEI data and train a model

1. Download the CMU-MultiModalSDK from github (see: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)
   and run the full example file (examples/mmdatasdk_examples/full_examples/process_mosei.py) for the CMU MOSEI dataset
    - it should have created a folder named "final_aligned"
2. Now create single folders with the labels and the matching features named like this: "aligned_covarep_data", "
   aligned_glove_vectors_data", "aligned_open_face_data"
3. Create a run configuration for running the src/train.py
4. In the script parameters set the hydra config you want to train, e.g. experiment=cmu_audio.yaml
   - to run the trimodal model, all unimodal models have to be run before
5. press play :)
    - there will be a logging folder generated in the logs/train folder with a timestamp. In it there is a "metrics.csv"
      with the logged data per step in each epoch 