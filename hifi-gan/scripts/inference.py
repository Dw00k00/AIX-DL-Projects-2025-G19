import torch
import sys
import numpy as np
import soundfile as sf
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models import Generator
from env import AttrDict
import json


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))



# -----------------------
#  CONFIGURATION
# -----------------------
CKPT_PATH = "cp_hifigan_refined/g_00000459"
CONFIG_PATH = "config_v1.json"
#MEL_PATH = "data/mels_refined/song2.npy"   # <- 여기 원하는 refined mel
MEL_PATH = "data/mels_refined/song2.npy"
OUTPUT_PATH = "outputs/song2_gen.wav"
# -----------------------


def load_hifigan_model(ckpt_path, config_path, device):
    # Load config
    with open(config_path, "r") as f:
        h = AttrDict(json.load(f))

    # Load model
    model = Generator(h).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict["generator"])
    model.eval()

    print(f"[Load] Loaded HiFi-GAN checkpoint: {ckpt_path}")
    return model, h


'''
def inference(mel_path, generator, h, device, out_path):
    mel = np.load(mel_path)
    mel = torch.FloatTensor(mel).unsqueeze(0).to(device)  # (1, n_mels, T)

  


    with torch.no_grad():
        audio = generator(mel).squeeze().cpu().numpy()

    sf.write(out_path, audio, h.sampling_rate)
    print(f"[Done] Saved output wav → {out_path}")

'''

def inference(mel_path, generator, h, device, out_path):
    mel = np.load(mel_path)                    # (n_mels, T)
    mel = torch.FloatTensor(mel).unsqueeze(0).to(device)  # (1, n_mels, T)

    T = mel.shape[-1]
    expected_len = T * h.hop_size             # mel 길이 기준 예상 샘플 수

    with torch.no_grad():
        audio = generator(mel).squeeze().cpu().numpy()    # (samples,)

    # 길이 보정: mel 기반 expected_len 에 맞추기
    if len(audio) > expected_len:
        audio = audio[:expected_len]
    elif len(audio) < expected_len:
        pad = expected_len - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")

    print(f"[Info] mel T={T}, hop={h.hop_size} → expected_len={expected_len}")
    print(f"[Info] final audio len={len(audio)} (samples)")

    sf.write(out_path, audio, h.sampling_rate)
    print(f"[Done] Saved output wav → {out_path}")


if __name__ == "__main__":
    device = torch.device("cpu")  

    generator, h = load_hifigan_model(CKPT_PATH, CONFIG_PATH, device)

    inference(
        MEL_PATH,
        generator,
        h,
        device,
        OUTPUT_PATH
    )

