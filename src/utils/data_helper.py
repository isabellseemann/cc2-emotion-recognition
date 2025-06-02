import numpy as np
import torch


def get_vector_with_only_dominant_emotion(label):
    max_value = np.max(label)  # Get the maximum value
    dominant_indices = np.where(label == max_value)[1]  # Get indices of the dominant values
    # Handle multiple dominant values (if tie occurs)
    if len(dominant_indices) > 1:
        selected_index = dominant_indices[len(dominant_indices) - 1]  # Randomly pick one of the indices
    else:
        selected_index = dominant_indices[0]  # Only one dominant value
    # Update the array to keep only the dominant value
    max_vector = np.zeros_like(label)  # Reset the array to zeros or default values
    max_vector[0][selected_index] = max_value
    return max_vector


def get_dominant_labels(labels):
    dominant_labels_train = []
    for count, label in enumerate(labels):
        max_vector = get_vector_with_only_dominant_emotion(label)
        label = max_vector
        dominant_labels_train.append(label)

    dominant_labels_train = np.array(dominant_labels_train)
    return dominant_labels_train


def cut_data(dominant_labels_train, open_face_data):
    happiness_counter = 0
    sadness_counter = 0
    #we used the commented out code to check if cutting the data more would improve the models performance
    # anger_counter = 0
    # disgust_counter = 0
    # surprise_counter = 0
    train_labels = []
    train_features = []
    for label, feature in zip(dominant_labels_train, open_face_data):
        max_value = np.max(label)
        indices_of_max = np.where(label == max_value)[1]
        max_index = indices_of_max[0]
        if max_index == 0:
            if happiness_counter == 2000:
                continue
            happiness_counter += 1
        if max_index == 1:
            if sadness_counter == 2000:
                continue
            sadness_counter += 1
        # if max_index == 2:
        #     if anger_counter == 2000:
        #         continue
        #     anger_counter += 1
        # if max_index == 3:
        #     if disgust_counter == 2000:
        #         continue
        #     disgust_counter += 1
        # if max_index == 4:
        #     if surprise_counter == 2000:
        #         continue
        #     surprise_counter += 1
        train_labels.append(label)
        train_features.append(feature)
    return train_labels, train_features


def get_text_train_data(glove_vec_tensors):
    dominant_labels = get_dominant_labels(glove_vec_tensors[0].get("All Labels"))
    train_labels, train_features = cut_data(dominant_labels, glove_vec_tensors[0].get("glove_vectors"))
    return train_labels, train_features


def get_text_val_data(glove_vec_tensors):
    dominant_labels = get_dominant_labels(glove_vec_tensors[1].get("All Labels"))
    val_labels, val_features = cut_data(dominant_labels, glove_vec_tensors[1].get("glove_vectors"))
    return val_labels, val_features


def get_text_test_data(glove_vec_tensors):
    dominant_labels = get_dominant_labels(glove_vec_tensors[2].get("All Labels"))
    test_labels, test_features = cut_data(dominant_labels, glove_vec_tensors[2].get("glove_vectors"))
    return test_labels, test_features


def get_video_train_data(video_tensors):
    dominant_labels = get_dominant_labels(video_tensors[0].get("All Labels"))
    train_labels, train_features = cut_data(dominant_labels, video_tensors[0].get("OpenFace_2"))
    return train_labels, train_features


def get_video_val_data(video_tensors):
    dominant_labels = get_dominant_labels(video_tensors[1].get("All Labels"))
    val_labels, val_features = cut_data(dominant_labels, video_tensors[1].get("OpenFace_2"))
    return val_labels, val_features


def get_video_test_data(video_tensors):
    dominant_labels = get_dominant_labels(video_tensors[2].get("All Labels"))
    test_labels, test_features = cut_data(dominant_labels, video_tensors[2].get("OpenFace_2"))
    return test_labels, test_features


def get_audio_train_data(audio_tensors):
    dominant_labels = get_dominant_labels(audio_tensors[0].get("All Labels"))
    train_labels, train_features = cut_data(dominant_labels, audio_tensors[0].get("COVAREP"))
    return train_labels, train_features


def get_audio_val_data(audio_tensors):
    dominant_labels = get_dominant_labels(audio_tensors[1].get("All Labels"))
    val_labels, val_features = cut_data(dominant_labels, audio_tensors[1].get("COVAREP"))
    return val_labels, val_features


def get_audio_test_data(audio_tensors):
    dominant_labels = get_dominant_labels(audio_tensors[2].get("All Labels"))
    test_labels, test_features = cut_data(dominant_labels, audio_tensors[2].get("COVAREP"))
    return test_labels, test_features


def get_trimodal_train_data(video_tensors, audio_tensors, glove_vec_tensors):
    train_text_labels, train_text_features = get_text_train_data(glove_vec_tensors)
    train_video_labels, train_video_features = get_video_train_data(video_tensors)
    train_audio_labels, train_audio_features = get_audio_train_data(audio_tensors)

    assert (
            torch.all(torch.tensor(train_text_labels) == torch.tensor(train_audio_labels))
            and torch.all(torch.tensor(train_text_labels) == torch.tensor(train_video_labels))
    ), "Labels not equal!!"

    return train_text_features, train_video_features, train_audio_features, train_audio_labels


def get_trimodal_val_data(video_tensors, audio_tensors, glove_vec_tensors):
    val_text_labels, val_text_features = get_text_val_data(glove_vec_tensors)
    val_video_labels, val_video_features = get_video_val_data(video_tensors)
    val_audio_labels, val_audio_features = get_audio_val_data(audio_tensors)

    assert (
            torch.all(torch.tensor(val_text_labels) == torch.tensor(val_audio_labels))
            and torch.all(torch.tensor(val_text_labels) == torch.tensor(val_video_labels))
    ), "Labels not equal!!"

    return val_text_features, val_video_features, val_audio_features, val_audio_labels


def get_trimodal_test_data(video_tensors, audio_tensors, glove_vec_tensors):
    test_text_labels, test_text_features = get_text_test_data(glove_vec_tensors)
    test_video_labels, test_video_features = get_video_test_data(video_tensors)
    test_audio_labels, test_audio_features = get_audio_test_data(audio_tensors)

    assert (
            torch.all(torch.tensor(test_text_labels) == torch.tensor(test_audio_labels))
            and torch.all(torch.tensor(test_text_labels) == torch.tensor(test_video_labels))
    ), "Labels not equal!!"

    return test_text_features, test_video_features, test_audio_features, test_audio_labels
