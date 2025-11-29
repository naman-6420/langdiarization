import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import PositionalEncoding, LayerNorm, EncoderBlock

class X_Base_E2E_LD(nn.Module):
    """
    This model uses statistical pooling (mean and standard deviation)
    over a window of 10 frames before feeding into the transformer.
    This corresponds to the W2V-ES model in the paper.
    """
    def __init__(self, feat_dim, d_k, d_v, d_ff, n_heads=4, dropout=0.1,
                 n_lang=3, max_seq_len=140,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(X_Base_E2E_LD, self).__init__()
        self.feat_dim = feat_dim
        self.device = device
        
        # Initial layers for the frame-level feature processing (the "xv" branch)
        self.dropout = nn.Dropout(p=dropout)
        # Input is 768 (mean) + 768 (std) = 1536
        self.fc1 = nn.Linear(1536, 3000) 
        self.bn1 = nn.BatchNorm1d(3000, momentum=0.1, affine=False)
        self.fc2 = nn.Linear(3000, feat_dim)
        self.bn2 = nn.BatchNorm1d(feat_dim, momentum=0.1, affine=False)
        self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.bn3 = nn.BatchNorm1d(feat_dim, momentum=0.1, affine=False)
        self.fc4 = nn.Linear(feat_dim, n_lang)

        # Transformer blocks for sequence-level processing
        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=feat_dim, device=device)
        self.layernorm2 = LayerNorm(feat_dim)
        self.attention_block1 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block3 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block4 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.output_fc = nn.Linear(feat_dim, n_lang)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, seq_len, atten_mask, eps=1e-5):
        # x has shape: [batch, sequence_length, feature_dim (768)]
        b, s, f = x.size()
        
        # Reshape to create windows of 10 frames
        x_win = x.view(b, s // 10, 10, f)
        
        # Statistical Pooling: Compute mean and std deviation across the 10-frame window
        xm = torch.mean(x_win, 2)
        xs = torch.std(x_win, 2)
        
        # Concatenate mean and std dev to get a 1536-dim vector
        pooled_x = torch.cat((xm, xs), 2)
        pooled_x = pooled_x.view(-1, 1536) # Reshape for the linear layers

        # --- Branch 1: CNN/XV-style classification head ---
        xv_branch = self.bn1(F.relu(self.fc1(pooled_x)))
        embedding = self.fc2(xv_branch) # This embedding is shared with the transformer
        xv_branch = self.bn2(F.relu(embedding))
        xv_branch = self.dropout(xv_branch)
        xv_branch = self.bn3(F.relu(self.fc3(xv_branch)))
        xv_branch = self.dropout(xv_branch)
        cnn_output = self.fc4(xv_branch)

        # --- Branch 2: Transformer for sequence modeling ---
        embedding = embedding.view(b, s // 10, self.feat_dim) # Reshape embedding back to sequence
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        output, _ = self.attention_block3(output, atten_mask)
        output, _ = self.attention_block4(output, atten_mask)
        # Unconventional sigmoid before CrossEntropyLoss, but preserved from original repo
        output = self.sigmoid(self.output_fc(output))

        return output, cnn_output

class X_Attention_E2E_LD(nn.Module):
    """
    This model uses attention-based pooling over a window of 10 frames
    before feeding into the transformer.
    This corresponds to the W2V-EA model in the paper.
    """
    def __init__(self, feat_dim, d_k, d_v, d_ff, n_heads=4, dropout=0.1,
                 n_lang=3, max_seq_len=140,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(X_Attention_E2E_LD, self).__init__()
        self.feat_dim = feat_dim
        self.device = device

        # BUGFIX: Changed input dimension from 99 to 768 to match Wav2Vec features
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 768)
        self.linear3 = nn.Linear(768, 1)      # Computes attention score for each frame
        self.linear4 = nn.Linear(768, 1536)   # Projects the pooled vector to 1536 dim

        # Subsequent layers are the same as the base model
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(1536, 3000) 
        self.bn1 = nn.BatchNorm1d(3000, momentum=0.1, affine=False)
        self.fc2 = nn.Linear(3000, feat_dim)
        self.bn2 = nn.BatchNorm1d(feat_dim, momentum=0.1, affine=False)
        self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.bn3 = nn.BatchNorm1d(feat_dim, momentum=0.1, affine=False)
        self.fc4 = nn.Linear(feat_dim, n_lang)

        # Transformer blocks
        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=feat_dim, device=device)
        self.layernorm2 = LayerNorm(feat_dim)
        self.attention_block1 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block3 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block4 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.output_fc = nn.Linear(feat_dim, n_lang)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, seq_len, atten_mask, eps=1e-5):
        # x has shape: [batch, sequence_length, feature_dim (768)]
        b, s, f = x.size()
        
        # Reshape to process all frames independently at first
        x_flat = x.view(-1, f) # a.k.a. x.view(b*s, 768)
        
        # --- Attention Pooling Logic ---
        # Pass frames through linear layers to get bottleneck features (H vectors)
        h_vectors = F.relu(self.linear1(x_flat))
        h_vectors = F.relu(self.linear2(h_vectors))
        
        # Calculate attention scores (weights) for each frame
        attn_scores = F.relu(self.linear3(h_vectors)) # Shape: [b*s, 1]
        
        # Reshape scores into windows of 10 to apply softmax
        attn_scores_win = attn_scores.view(b * (s // 10), 10, 1)
        attn_weights = F.softmax(attn_scores_win, dim=1) # Softmax over the 10 frames in the window

        # Reshape bottleneck vectors into windows of 10
        h_vectors_win = h_vectors.view(b * (s // 10), 10, f)

        # Apply attention weights: weighted sum of bottleneck vectors in each window
        weighted_vectors = h_vectors_win * attn_weights
        pooled_x = torch.sum(weighted_vectors, dim=1) # Shape: [b * (s/10), 768]
        
        # Project the 768-dim attended vector to 1536 to match the next layer's input
        pooled_x = F.relu(self.linear4(pooled_x))

        # --- Branch 1: CNN/XV-style classification head (same as base model from here) ---
        xv_branch = self.bn1(F.relu(self.fc1(pooled_x)))
        embedding = self.fc2(xv_branch)
        xv_branch = self.bn2(F.relu(embedding))
        xv_branch = self.dropout(xv_branch)
        xv_branch = self.bn3(F.relu(self.fc3(xv_branch)))
        xv_branch = self.dropout(xv_branch)
        cnn_output = self.fc4(xv_branch)

        # --- Branch 2: Transformer for sequence modeling ---
        embedding = embedding.view(b, s // 10, self.feat_dim)
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        output, _ = self.attention_block3(output, atten_mask)
        output, _ = self.attention_block4(output, atten_mask)
        output = self.sigmoid(self.output_fc(output))
        
        return output, cnn_output

