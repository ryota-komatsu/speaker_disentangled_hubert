import random
from pathlib import Path

import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x: torch.Tensor, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y: torch.FloatTensor,
    n_fft: int = 400,
    num_mels: int = 80,
    sampling_rate: int = 16000,
    hop_size: int = 320,
    fmin=0,
    fmax=8000,
    center: bool = False,
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(n_fft).to(y.device)

    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        window=hann_window[str(y.device)],
        center=center,
        onesided=True,
        return_complex=True,
    )

    spec = spec.abs()
    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = dynamic_range_compression_torch(spec)

    return spec


class MelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_wavs_dir,
        input_mels_dir,
        segment_size: int,
        n_fft: int = 400,
        hop_size: int = 320,
        split: bool = True,
        ext_audio: str = ".wav",
    ):
        self.segment_size = segment_size
        self.hop_size = hop_size
        self.split = split
        self.frames_per_seg = (segment_size - n_fft) // hop_size + 1
        self.mel_pad_value = dynamic_range_compression_torch(torch.tensor(0))

        self.wav_paths = []
        self.mel_paths = []

        input_wavs_dir = Path(input_wavs_dir)
        input_mels_dir = Path(input_mels_dir)

        wav_paths = input_wavs_dir.glob("**/*" + ext_audio)

        for wav_path in wav_paths:
            mel_path = input_mels_dir / wav_path.relative_to(input_wavs_dir).with_suffix(".pt")
            wav_path = str(wav_path)

            self.wav_paths.append(wav_path)
            self.mel_paths.append(mel_path)

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index: int):
        wav_path = self.wav_paths[index]
        mel_path = self.mel_paths[index]

        audio, sr = torchaudio.load(wav_path)
        audio = audio / audio.abs().max() * 0.95
        audio = audio.squeeze(0)

        mel = torch.load(mel_path, map_location="cpu", weights_only=True)  # (1, len, 80)
        mel = mel.squeeze(0)  # (len, 80)
        mel = mel.transpose(0, 1)  # (80, len)

        mask = torch.ones_like(mel, dtype=torch.bool)

        if self.split:
            diff = mel.size(1) - self.frames_per_seg
            if diff > 0:
                mel_start = random.randrange(diff)
                mel = mel[:, mel_start : mel_start + self.frames_per_seg]
                audio = audio[mel_start * self.hop_size : mel_start * self.hop_size + self.segment_size]
                mask = mask[:, mel_start : mel_start + self.frames_per_seg]
            else:
                mel = torch.nn.functional.pad(mel, (0, -diff), value=self.mel_pad_value)
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(0)))
                mask = torch.nn.functional.pad(mask, (0, -diff))

        return mel, audio, mask
