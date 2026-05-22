import re
from threading import Thread
from typing import Tuple

import gradio as gr
import numpy as np
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.flow_matching import FlowMatchingWithBigVGan
from src.s5hubert import SylRegForSyllableDiscovery

device = "cuda" if torch.cuda.is_available() else "cpu"

# download pretrained models from hugging face hub
encoder = SylRegForSyllableDiscovery.from_pretrained("ryota-komatsu/SylReg-Distill", device_map=device)
decoder = FlowMatchingWithBigVGan.from_pretrained("ryota-komatsu/SylReg-Decoder", device_map=device)
speechlm = AutoModelForCausalLM.from_pretrained("/path/to/speechLM", device_map="cuda", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("/path/to/speechLM")


def synthesize(audio: str):
    # load a waveform
    waveform, sr = torchaudio.load(audio)
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # encode a waveform into syllabic units
    units = encoder(waveform.to(encoder.device))[0]["units"]  # [3950, 67, ..., 503]

    # speech language modeling
    text = "".join(f"<{unit}>" for unit in units)
    input_ids = tokenizer(text, padding=True, return_tensors="pt").input_ids.to(speechlm.device)
    generated_ids = speechlm.generate(input_ids=input_ids, do_sample=True, temperature=0.8)[0]
    units = tokenizer.decode(generated_ids)
    units = torch.tensor([int(unit) for unit in re.findall(r"<(\d+)>", units)], device=decoder.device)

    # unit-to-speech synthesis
    outputs = decoder(units.unsqueeze(0))
    generated_speech = outputs.waveform.squeeze(0).cpu().numpy()

    return 16000, generated_speech


if __name__ == "__main__":
    with gr.Blocks(title="Speech Resynthesis") as demo:
        with gr.Row():
            audio_in = gr.Audio(type="filepath", label="Original speech")

        with gr.Row():
            btn = gr.Button("Resynthesize")
            audio_out = gr.Audio(label="Generated speech", streaming=True, autoplay=True)

        btn.click(synthesize, inputs=audio_in, outputs=audio_out)

    demo.launch()
