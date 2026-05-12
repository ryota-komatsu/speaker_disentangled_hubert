import warnings

import matplotlib.pyplot as plt
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..s5hubert import SylRegForSyllableDiscovery

warnings.simplefilter("ignore", UserWarning)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


@torch.inference_mode()
def analyze(
    audio_filepath: str,
    enc_name: str,
    lm_name: str,
):
    encoder = SylRegForSyllableDiscovery.from_pretrained(enc_name, device_map="cuda", dtype="auto")
    speechlm = AutoModelForCausalLM.from_pretrained(lm_name, device_map="cuda", dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(lm_name)

    # load a waveform
    waveform, sr = torchaudio.load(audio_filepath)
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # encode a waveform into syllabic units
    outputs = encoder(waveform.to(encoder.device))
    units = outputs[0]["units"]  # [3950, 67, ..., 503]

    # speech language modeling
    input_units = "".join(f"<{unit}>" for unit in units)
    input_ids = tokenizer(input_units, padding=True, return_tensors="pt").input_ids.to(speechlm.device)
    speech_hidden_states = speechlm(input_ids, output_hidden_states=True).hidden_states

    for layer_index, speech_h in enumerate(speech_hidden_states):
        speech_h = torch.nn.functional.normalize(speech_h.squeeze(0), dim=1)
        similarity = speech_h @ speech_h.T

        plt.figure(figsize=[25.6, 19.2])
        plt.imshow(similarity.cpu().numpy(), cmap="bwr", vmin=-1, vmax=1)
        plt.xlabel("Speech", fontsize=16)
        plt.savefig(f"similarity{layer_index:02}.png", bbox_inches="tight")
