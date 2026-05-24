import re

import gradio as gr
import torch
import torchaudio
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)

from src.flow_matching import FlowMatchingWithBigVGan
from src.s5hubert import SylRegForSyllableDiscovery

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# download pretrained models from hugging face hub
encoder = SylRegForSyllableDiscovery.from_pretrained("ryota-komatsu/SylReg-Distill", device_map=device)
decoder = FlowMatchingWithBigVGan.from_pretrained("ryota-komatsu/SylReg-Decoder", device_map=device)
speechlm = AutoModelForCausalLM.from_pretrained("/path/to/speechLM", device_map="cuda", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("/path/to/speechLM")

asr = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
pipe = pipeline(
    "automatic-speech-recognition",
    model=asr,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
)


def main(audio: str):
    # load a waveform
    waveform, sr = torchaudio.load(audio)
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # encode a waveform into syllabic units
    units = encoder(waveform.to(encoder.device))[0]["units"]  # [3950, 67, ..., 503]

    # speech language modeling
    text = "".join(f"<{unit}>" for unit in units[:-1])
    input_ids = tokenizer(text, padding=True, return_tensors="pt").input_ids.to(speechlm.device)
    generated_ids = speechlm.generate(input_ids=input_ids, do_sample=True, temperature=0.8)[0]
    units = tokenizer.decode(generated_ids)
    units = torch.tensor([int(unit) for unit in re.findall(r"<(\d+)>", units)], device=decoder.device)

    # unit-to-speech synthesis
    outputs = decoder(units.unsqueeze(0))
    generated_speech = outputs.waveform.squeeze(0).cpu().numpy()

    generated_text = pipe(generated_speech, generate_kwargs={"language": "english"}, return_timestamps=True)["text"]

    return (16000, generated_speech), generated_text


if __name__ == "__main__":
    with gr.Blocks(title="Speech Resynthesis") as demo:
        with gr.Row():
            audio_in = gr.Audio(type="filepath", label="Original speech")

        with gr.Row():
            btn = gr.Button("Generate")
            audio_out = gr.Audio(label="Generated speech", streaming=True, autoplay=True)

        with gr.Row():
            text_out = gr.Textbox(label="Transcript")

        btn.click(main, inputs=audio_in, outputs=[audio_out, text_out])

    demo.launch()
