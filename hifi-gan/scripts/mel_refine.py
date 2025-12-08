"""
mel_refine.py

기능:
- data/mels_original 안의 멜(.npy) 파일들을 읽어서
- 멜 refiner(현재는 기본적으로 "그대로 통과" = identity) 를 통과시킨 뒤
- data/mels_refined 에 저장한다.

나중에 실제 Diffusion Mel Refiner 모델을 만들면
`load_mel_refiner()` 부분만 수정해서 그대로 사용하면 된다.

사용 예:
    python mel_refine.py \
        --input_mel_dir data/mels_original \
        --output_mel_dir data/mels_refined
"""

import argparse
from pathlib import Path
from typing import Optional
import torch.nn.functional as F

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ==========================
# 1. Refiner 로더
# ==========================


class SimpleMelRefiner(nn.Module):
    """
    아주 간단한 멜 refiner 예시:
    - 시간 축으로 살짝 smoothing
    - 주파수 대역별로 약간 다른 gain을 줘서 톤 변화
    """
    def __init__(self, num_mels: int = 80, smooth_kernel: int = 5):
        super().__init__()
        self.num_mels = num_mels
        self.smooth_kernel = smooth_kernel

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, 80, T)
        B, C, T = mel.shape
        x = mel

        # 1) 시간 방향 average pooling으로 부드럽게 (kernel=5)
        x_smooth = F.avg_pool1d(
            x,
            kernel_size=self.smooth_kernel,
            stride=1,
            padding=self.smooth_kernel // 2,
        )

        # 2) 주파수(멜 index)에 따라 살짝 다른 gain
        device = mel.device
        freq_pos = torch.linspace(0.0, 1.0, steps=C, device=device).view(1, C, 1)
        # 예시: 중역대(0.3~0.7) 살짝 올리고, 최저/최고는 조금 줄이기
        freq_weight = 1.0 + 0.25 * torch.sin(torch.pi * freq_pos) - 0.15 * freq_pos

        x_refined = 0.7 * x + 0.3 * x_smooth
        x_refined = x_refined * freq_weight

        return x_refined


def load_mel_refiner(ckpt_path: Optional[Path], device: torch.device) -> nn.Module:
    """
    멜 refiner 모델 로드.
    지금은 SimpleMelRefiner 사용 (파이프라인/보고서용 데모).
    """
    model = SimpleMelRefiner(num_mels=80)

    if ckpt_path is not None:
        if ckpt_path.is_file():
            print(f"[Warn] ckpt {ckpt_path} 는 아직 로딩 로직이 구현되지 않았습니다.")
            print("       load_mel_refiner() 안에 실제 diffusion 모델 로딩 코드를 넣어주세요.")
        else:
            print(f"[Warn] ckpt_path={ckpt_path} 가 파일이 아님. SimpleMelRefiner로 진행합니다.")
    else:
        print("[Info] ckpt_path 없음 → SimpleMelRefiner 사용 (입력에 부드러운 변형만 가함).")

    model.to(device)
    model.eval()
    return model

# ==========================
# 2. 멜 로딩 / 저장
# ==========================

def load_mel_npy(path: Path, expected_mels: int = 80) -> np.ndarray:
    """
    .npy 멜 파일 로드.
    - shape 이 (80, T) 또는 (T, 80)인 경우 모두 대응.
    """
    mel = np.load(path)  # float32 [?, ?]
    if mel.ndim != 2:
        raise ValueError(f"[Error] {path} 멜 shape={mel.shape}, 2D 가 아님.")

    h, w = mel.shape
    if h == expected_mels:
        # (80, T) -> 그대로 사용
        return mel.astype(np.float32)
    elif w == expected_mels:
        # (T, 80) -> (80, T)로 transpose
        return mel.T.astype(np.float32)
    else:
        raise ValueError(
            f"[Error] {path} 멜 shape={mel.shape}, 어느 축도 {expected_mels} 이 아님."
        )


def save_mel_npy(path: Path, mel: np.ndarray):
    """
    멜을 .npy로 저장. shape 은 (80, T) 로 통일해서 저장.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mel.astype(np.float32))


# ==========================
# 3. 전체 처리 루프
# ==========================

def refine_all_mels(
    input_mel_dir: Path,
    output_mel_dir: Path,
    ckpt_path: Optional[Path] = None,
    num_mels: int = 80,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1) 모델 로드
    refiner = load_mel_refiner(ckpt_path, device)

    # 2) 입력 멜 리스트
    input_mel_dir = input_mel_dir.resolve()
    output_mel_dir = output_mel_dir.resolve()

    mel_files = sorted(input_mel_dir.glob("*.npy"))
    if len(mel_files) == 0:
        print(f"[Warn] 입력 디렉토리 {input_mel_dir} 에 .npy 멜 파일이 없습니다.")
        return

    print(f"[Info] 멜 파일 개수: {len(mel_files)}")
    print(f"[Info] 입력: {input_mel_dir}")
    print(f"[Info] 출력: {output_mel_dir}")

    # 3) 루프
    with torch.no_grad():
        for mel_path in tqdm(mel_files, desc="Refining mels"):
            # (1) 멜 로드
            mel_np = load_mel_npy(mel_path, expected_mels=num_mels)  # (80, T)

            # (2) torch tensor (1, 80, T)
            mel_tensor = torch.from_numpy(mel_np).unsqueeze(0).to(device)

            # (3) refiner 통과
            refined = refiner(mel_tensor)  # (1, 80, T)
            refined_np = refined.squeeze(0).cpu().numpy()  # (80, T)

            # (4) 저장: 파일명은 그대로, 디렉토리만 변경
            out_path = output_mel_dir / mel_path.name
            save_mel_npy(out_path, refined_np)

    print("[Done] 모든 멜 refined + 저장 완료.")


# ==========================
# 4. CLI
# ==========================

def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Mel Refiner - STEP 1")
    parser.add_argument(
        "--input_mel_dir",
        type=str,
        default="data/mels_original",
        help="원본 멜(.npy) 디렉토리 (기본: data/mels_original)",
    )
    parser.add_argument(
        "--output_mel_dir",
        type=str,
        default="data/mels_refined",
        help="refined 멜(.npy) 저장 디렉토리 (기본: data/mels_refined)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="멜 refiner checkpoint 경로 (현재는 사용 안 함, 나중에 확장용)",
    )
    parser.add_argument(
        "--num_mels",
        type=int,
        default=80,
        help="멜 차원 수 (HiFi-GAN config_v1.json 기준 80)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    refine_all_mels(
        input_mel_dir=Path(args.input_mel_dir),
        output_mel_dir=Path(args.output_mel_dir),
        ckpt_path=Path(args.ckpt) if args.ckpt is not None else None,
        num_mels=args.num_mels,
    )


if __name__ == "__main__":
    main()

