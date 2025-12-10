import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)

    # 스테레오(2채널)인 경우 → 모노로 변환
    if data.ndim == 2:
        # axis=1 기준으로 두 채널 평균
        data = data.mean(axis=1)

    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}



import torch
import torch.nn.functional as F
import librosa
import numpy as np

# 파일 맨 위쪽에 이미 있을 가능성 큼
mel_basis = {}
hann_window = {}

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    center: bool = False,   # ← 추가

):
    """
    HiFi-GAN 스타일 mel-spectrogram 계산 (최신 PyTorch/Librosa 호환 버전)

    입력:
        y: (T,) 또는 (1, T) 실수 파형
    출력:
        mel: (num_mels, T_frames)
    """
    global mel_basis, hann_window

    # --- 1) 입력 형태 정리: (T,) 로 만들기 ---
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    y = y.float()

    # (1, T) 또는 (B, T) 인 경우는 첫 배치만 사용 (wav_to_mel에서는 (1, T)만 들어옴)
    if y.dim() == 2:
        y = y[0]  # (T,)

    if y.dim() != 1:
        raise ValueError(f"mel_spectrogram: 예상치 못한 입력 shape {y.shape}, 1D 파형이어야 합니다.")

    # --- 2) [-1, 1] 확인 (HiFi-GAN 관례용) ---
    if torch.min(y) < -1.0 or torch.max(y) > 1.0:
        print(f"[Warn] mel_spectrogram: wave min={torch.min(y)}, max={torch.max(y)}")

    # --- 3) 패딩 (1D zero-padding) ---
    pad = int((n_fft - hop_size) // 2)
    y = F.pad(y, (pad, pad), mode="constant", value=0.0)  # (T_padded,)

    # --- 4) Hann window 캐시 ---
    device = y.device
    win_key = f"{device}_{win_size}"
    if win_key not in hann_window:
        hann_window[win_key] = torch.hann_window(win_size).to(device)
    window = hann_window[win_key]

    # --- 5) STFT (최신 PyTorch: return_complex=True 필수) ---
    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )  # (freq, frames) complex

    spec = spec.abs()  # magnitude: (freq, frames)

    # --- 6) mel basis 캐시 ---
    mel_key = f"{sampling_rate}_{device}"
    if mel_key not in mel_basis:
        mel = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        mel_basis[mel_key] = torch.from_numpy(mel).float().to(device)

    mel_filter = mel_basis[mel_key]  # (num_mels, freq)

    # --- 7) mel 변환 ---
    # mel_filter (num_mels, freq) x spec (freq, frames) -> (num_mels, frames)
    mel_spec = torch.matmul(mel_filter, spec)

    # --- 8) dynamic range compression ---
    mel_spec = dynamic_range_compression(mel_spec)

    return mel_spec  # (num_mels, frames)

def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
