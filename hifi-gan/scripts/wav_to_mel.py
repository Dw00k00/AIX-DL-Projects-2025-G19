

from __future__ import annotations
import librosa  
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav  


def parse_args():
    parser = argparse.ArgumentParser(description="Wav -> Mel (HiFi-GAN style)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="HiFi-GAN config json (ex: config_v1.json)",
    )
    parser.add_argument(
        "--input_wavs_dir",
        type=str,
        required=True,
        help="원본 wav 디렉토리 (ex: data/wavs_original)",
    )
    parser.add_argument(
        "--output_mel_dir",
        type=str,
        required=True,
        help="추출된 mel(.npy) 저장 디렉토리 (ex: data/mels_original)",
    )
    return parser.parse_args()


def load_hifigan_config(config_path: Path) -> AttrDict:
    with open(config_path, "r") as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)  # env.AttrDict: dict -> attribute access 
    return h


def wav_to_mel_file(
    wav_path: Path,
    mel_out_path: Path,
    h: AttrDict,
    device: torch.device,
):
    # 1) wav 로드
    wav, sr = load_wav(str(wav_path))

    # ✅ (1) 스테레오 → 모노 변환
    if wav.ndim > 1:
        print(f"[Info] {wav_path.name}: wav shape {wav.shape}, "
              f"채널이 여러 개라 첫 채널만 사용합니다.")
        wav = wav[:, 0]   # 또는 wav.mean(axis=1)

    # ✅ (2) 샘플레이트 맞추기 (48k → 22.05k)
    if sr != h.sampling_rate:
        print(f"[Info] {wav_path.name}: resample {sr} -> {h.sampling_rate}")
        wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=h.sampling_rate)
        sr = h.sampling_rate

    # ✅ (3) [-1, 1] 정규화
    wav = wav / MAX_WAV_VALUE

    # ✅ (4) torch tensor (1D 유지)
    wav_tensor = torch.from_numpy(wav).float().to(device)  # (T,)

    # ✅ (5) mel-spectrogram 계산
    mel = mel_spectrogram(
        wav_tensor,
        n_fft=h.n_fft,
        num_mels=h.num_mels,
        sampling_rate=h.sampling_rate,
        hop_size=h.hop_size,
        win_size=h.win_size,
        fmin=h.fmin,
        fmax=h.fmax,
    )  # (num_mels, frames)

    # ✅ (6) numpy 저장
    mel_np = mel.cpu().numpy().astype(np.float32)
    mel_out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mel_out_path, mel_np)



def main():
    args = parse_args()

    config_path = Path(args.config)
    input_wavs_dir = Path(args.input_wavs_dir)
    output_mel_dir = Path(args.output_mel_dir)

    h = load_hifigan_config(config_path)
    print(f"[Config] sampling_rate={h.sampling_rate}, num_mels={h.num_mels}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    wav_files = sorted(
        [p for p in input_wavs_dir.glob("*.wav") if p.is_file()]
    )
    if len(wav_files) == 0:
        print(f"[Warn] {input_wavs_dir} 에 .wav 파일이 없습니다.")
        return

    print(f"[Info] wav 파일 개수: {len(wav_files)}")
    print(f"[Info] 입력: {input_wavs_dir}")
    print(f"[Info] 출력: {output_mel_dir}")

    for wav_path in wav_files:
        out_name = wav_path.stem + ".npy"
        mel_out_path = output_mel_dir / out_name

        print(f"[Proc] {wav_path.name} -> {out_name}")
        wav_to_mel_file(
            wav_path=wav_path,
            mel_out_path=mel_out_path,
            h=h,
            device=device,
        )

    print("[Done] 모든 wav -> mel(.npy) 변환 완료.")


if __name__ == "__main__":
    main()
