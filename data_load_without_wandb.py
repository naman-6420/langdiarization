import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils

def collate_fn_cnn_atten(batch):
    """
    Custom collate function to handle variable length sequences.
    Sorts by feature length, pads features, and prepares flat label tensors for both losses.
    """
    # Sort the batch by the original feature length (index 3) in descending order
    batch.sort(key=lambda x: x[3], reverse=True)
    seq, label, filename, feat_lens, label_lens = zip(*batch)

    # Pad the feature sequences
    padded_features = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)

    # Concatenate all label tensors into a single flat tensor.
    # This will be used for both the transformer loss and the XV loss.
    flat_labels = torch.cat(label, dim=0)

    # Note: Returning feat_lens and label_lens for potential debugging, though not strictly used in the loop now.
    # The training scripts now expect 5 items to unpack, so we keep the signature.
    return padded_features, flat_labels, flat_labels, feat_lens, label_lens

class RawFeatures(data.Dataset):
    def __init__(self, txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.feature_list = [i.split()[0] for i in lines]
            self.label_list = [i.split()[-1] for i in lines]

    def __getitem__(self, index):
        feature_path = self.feature_list[index]
        feature = torch.from_numpy(np.load(feature_path, allow_pickle=True))
        # Features are loaded with shape [1, num_frames, 768], so we squeeze it
        if feature.dim() == 3:
            feature = feature.squeeze(0)
        
        filename = feature_path.split('/')[-1]

        # --- BUG FIX: Ensure label length matches downsampled feature length ---
        # The model architecture downsamples the temporal dimension of the features by a factor of 10.
        # The ground truth label sequence must have a length that matches this new, shorter dimension.
        expected_label_len = feature.shape[0] // 10
        label_str = self.label_list[index]

        # Trim the label string to the expected length. This is the critical fix for the ValueError.
        trimmed_label_str = label_str[:expected_label_len]
        
        label = []
        # Convert character labels to numeric format based on user request
        for x in trimmed_label_str:
            if x == 'S':
                label.append(0)
            elif x == 'E':
                label.append(2)
            else:  # Assume any other character ('G', 'T', etc.) is the primary language
                label.append(1)

        return feature, torch.LongTensor(label), filename, feature.shape[0], len(label)

    def __len__(self):
        return len(self.label_list)

def get_atten_mask(seq_lens, batch_size):
    """Creates an attention mask for a batch of sequences of varying lengths."""
    max_len = seq_lens[0] # Assumes sequences are sorted by length in descending order
    atten_mask = torch.ones([batch_size, max_len, max_len])
    for i in range(batch_size):
        length = seq_lens[i]
        atten_mask[i, :length, :length] = 0
    return atten_mask.bool()

