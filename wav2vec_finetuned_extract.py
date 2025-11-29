import os
import glob
from fairseq import checkpoint_utils
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


import sys
sys.modules["wandb"] = None

# -------------------- DEVICE SETUP -------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- DATASET CLASS -------------------- #
class MS_Dataset(Dataset):
    def __init__(self, file_list):
        self.train = file_list
        self.utter = []
        self.fname = []
        for path in self.train:
            fname = os.path.basename(path)
            data, fs = sf.read(path)
            self.utter.append(data)
            self.fname.append(fname)

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.utter[index])), self.fname[index]

    def __len__(self):
        return len(self.utter)


# -------------------- FEATURE EXTRACTION FUNCTION -------------------- #
def extract_features(language, split, model_path, base_data_dir, base_feature_dir):
    """
    language        : e.g. "GUE", "TAM", "TEL"
    split           : "Train" / "Dev" / "Test"
    model_path      : path to the fine-tuned checkpoint
    base_data_dir   : path to data_raw/<LANG>
    base_feature_dir: where to save extracted features
    """

    print(f"\n🔹 Starting feature extraction for {language} | {split} split")

    # Paths
    audio_files = glob.glob(os.path.join(base_data_dir, split, "Audio", "*.wav"))
    transcript_path = os.path.join(base_data_dir, split, "Transcription_LT_Sequence_Frame_Level_200_actual.tsv")
    save_dir = os.path.join(base_feature_dir, language, "finetuned", split.lower())
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    print("Loading Wav2Vec2 model...")
    # Corrected import and function call
    model, cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path])
    model = model[0].to(device)
    mod = model.w2v_encoder.w2v_model
    model.eval()

    # Load labels
    labels_dict = {}
    if os.path.exists(transcript_path):
        with open(transcript_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    labels_dict[parts[0]] = parts[1]
        print(f"Loaded {len(labels_dict)} transcript labels.")
    else:
        print(f"⚠ No transcript found for {language} {split}!")

    # Prepare dataset + dataloader
    dataset = MS_Dataset(audio_files)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    # Output TSV path
    tsv_path = os.path.join(base_feature_dir, language, "finetuned", f"WAVData_{split.upper()}.txt")
    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)

    # Extract features
    missing_labels = []
    with open(tsv_path, "w") as tsv_file:
        for _, (batch, fname) in enumerate(tqdm(dataloader, desc=f"Extracting {split} features")):
            audio_id = os.path.splitext(fname[0])[0]
            train_data = batch.to(dtype=torch.float).to(device)

            # Forward pass
            with torch.no_grad():
                z = mod.forward(train_data, mask=False, features_only=True)
            mynpy = z["x"].data.cpu().numpy()

            # Trim to nearest multiple of 10
            dim1 = mynpy.shape[1]
            res = (dim1 // 10) * 10
            mynpy = mynpy[:, :res, :]

            # Save features
            new_save_path = os.path.join(save_dir, audio_id + ".npy")
            np.save(new_save_path, mynpy)

            # Write TSV mapping
            if audio_id in labels_dict:
                tsv_file.write(new_save_path + "\t" + labels_dict[audio_id] + "\n")
            else:
                missing_labels.append(audio_id)

    # Missing label logs
    if missing_labels:
        log_path = os.path.join(base_feature_dir, language, "finetuned", f"missing_labels_{split}.log")
        with open(log_path, "w") as log_file:
            for m in missing_labels:
                log_file.write(m + "\n")
        print(f"⚠ Missing labels for {len(missing_labels)} files → Check: {log_path}")
    else:
        print("✅ All labels matched successfully!")

    print(f"✅ Completed feature extraction for {language} | {split} split")
    print(f"🔹 Features saved at: {save_dir}")
    print(f"🔹 TSV mapping saved at: {tsv_path}")


# -------------------- MAIN SCRIPT -------------------- #
if __name__ == "__main__":
    # Set paths
    
    base_dir = "/home/b22164/wd/interspeech_23/W2V-E2E-Language-Diarization-main"

    # Telugu (TEE)
    # model_path_tee = os.path.join(base_dir, "models/checkpoint_best_telegu.pt")
    # base_data_dir_tee = os.path.join(base_dir, "data_raw/TEE")
    base_feature_dir = os.path.join(base_dir, "features")

    # extract_features("TEE", "Train", model_path_tee, base_data_dir_tee, base_feature_dir)
    # extract_features("TEE", "Dev", model_path_tee, base_data_dir_tee, base_feature_dir)
    # extract_features("TEE", "Test", model_path_tee, base_data_dir_tee, base_feature_dir)

    # Tamil (TAE)
    model_path_tae = os.path.join(base_dir, "models/checkpoint_best_gujrati.pt")
    base_data_dir_tae = os.path.join(base_dir, "data_raw/GUE")

    extract_features("GUE", "Train", model_path_tae, base_data_dir_tae, base_feature_dir)
    extract_features("GUE", "Dev", model_path_tae, base_data_dir_tae, base_feature_dir)
    # extract_features("TAE", "Test", model_path_tae, base_data_dir_tae, base_feature_dir)

