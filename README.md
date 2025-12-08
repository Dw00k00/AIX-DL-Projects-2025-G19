# AIX-DL-Projects-2025-G19  
2025-2 AIX 기말 프로젝트  

---

## 🧑‍🎓 Members
- **이현우** (인공지능학과, 2025001712) — doctor0514@hanyang.ac.kr 
  *코드 작성, Diffusion Mel Refiner 구현, HiFi-GAN 파이프라인 구축, 분석 노트북 작성*
- **민동욱** (전자공학부, 2021017874) — mdu1009@hanyang.ac.kr  
  *코드 작성, mel 변환·전처리 파이프라인 구현, 결과 시각화·실험 지원*

---
Link
동영상 링크 :

ipynb 파일 뷰어 링크 :https://nbviewer.org/github/Dw00k00/AIX-DL-Projects-2025-G19/blob/main/notebooks/hifi_report.ipynb
# 🎵 딥러닝 기반 노래 음성 스타일 전이 (Singing Voice Conversion, SVC)

본 프로젝트는 **딥러닝을 이용한 노래 음성 스타일 전이(SVC)** 를 목표로 한다.  
특히, mel-spectrogram 기반 음성 표현을 바꾸는 “Diffusion Mel Refiner”를 구현하고,  
그 결과를 **HiFi-GAN vocoder**를 사용해 waveform으로 복원하는 전체 파이프라인을 구축하였다.

이 과정에서:

- 🔹 Refiner가 mel-spectrogram의 분포에 어떤 영향을 주는지  
- 🔹 변경된 mel을 HiFi-GAN이 어떻게 복원하는지  
- 🔹 전처리 mismatch가 vocoder collapse로 이어지는 과정  

을 실험과 분석을 통해 확인하였다.

세부 분석 및 그래프, 통계 결과는 `notebooks/hifi_report.ipynb` 에 자세히 정리되어 있다.

---

# 📁 프로젝트 구조
hifi-gan/
├── scripts/                     
│   ├── wav_to_mel.py        # STEP 1: WAV → MEL 변환
│   ├── mel_refine.py        # STEP 2: Diffusion Mel Refiner
│   └── inference.py         # STEP 3: MEL → HiFi-GAN vocoder
│
├── data/
│   ├── wavs_original/       # 입력: 원본 노래 WAV
│   ├── mels_original/       # 원본 mel-spectrogram
│   ├── mels_refined/        # Refiner 출력 mel
│   └── generated/           # Vocoder로 복원된 WAV
│
├── models.py                # HiFi-GAN Generator / Discriminator 정의
├── env.py                   # HiFi-GAN 설정 로더
│
└── notebooks/
    └── hifi_report.ipynb    # 분석용 Notebook


# 핵심 요약

본 프로젝트는 딥러닝 기반 singing-voice-conversion 파이프라인을 구축하는 데 목적이 있다.

Mel Refiner가 mel 분포를 조금만 바꿔도,
기존 HiFi-GAN vocoder는 학습 분포 mismatch로 인해 collapse 될 수 있다.

이를 통해 전처리 정의의 중요성과 모델 간 입력 호환성 문제를 실험적으로 확인하였다.

자세한 실험 결과와 시각화는 Jupyter Notebook에서 확인할 수 있다.

notebooks/hifi_report.ipynb — 전체 실험·분석 결과

scripts/ — 데이터 전처리 및 vocoder inference 코드

data/ — 실험 input/output 파일들 저장

---

---

# 🚀 실행 방법 (End-to-End)

```bash
cd hifi-gan
.\.venv\Scripts\activate

# 1) WAV → Mel 변환
python scripts/wav_to_mel.py --config config_v1.json ^
  --input_wavs_dir data/wavs_original ^
  --output_mel_dir data/mels_original

# 2) Mel Refiner 적용
python scripts/mel_refine.py ^
  --input_mel_dir data/mels_original ^
  --output_mel_dir data/mels_refined

# 3) Vocoder 복원
python scripts/inference.py


# 🏋️‍♂️ Optional: HiFi-GAN Vocoder Training (Refined Mel 기반 재학습)

본 프로젝트에서는 분석을 위해 HiFi-GAN의 기본 pre-trained 모델을 사용했지만,  
원한다면 **Refined Mel 분포에 맞춰 vocoder를 재학습(finetune)** 할 수도 있다.

이를 통해 mel 분포 mismatch 문제를 완화할 수 있으며,  
Refiner의 출력 특성에 더 적합한 vocoder를 얻을 수 있다.

## 🔧 준비물
- `data/wavs_original/` — 원본 WAV  
- `data/mels_refined/` — Refiner가 생성한 refined mel  
- `data/filelists/train.txt` — 학습용 파일 리스트  
- `data/filelists/val.txt` — 검증용 파일 리스트  
- `config_v1.json` — HiFi-GAN 학습 설정  
- `train.py` — HiFi-GAN 공식 training 스크립트  

---

## 🚀 학습 실행 명령어

아래 명령어 한 줄이면 refined mel 기반 HiFi-GAN 학습을 시작할 수 있다:

```bash
python train.py ^
  --config config_v1.json ^
  --input_wavs_dir data/wavs_original ^
  --input_mels_dir data/mels_refined ^
  --input_training_file data/filelists/train.txt ^
  --input_validation_file data/filelists/val.txt ^
  --checkpoint_path cp_hifigan_refined ^
  --checkpoint_interval 1 ^
  --stdout_interval 1
