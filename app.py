import re

import gradio as gr
import matplotlib.pyplot as plt
import torch
import torchaudio
from datasets import Audio, load_dataset
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

transform = torchaudio.transforms.MelSpectrogram(hop_length=320, n_mels=80, center=False).to(device)

dataset = load_dataset("fixie-ai/llama-questions", split="test")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.with_format("torch")


@torch.inference_mode()
def main(audio: str, temperature: float):
    # load a waveform
    input_values, sr = torchaudio.load(audio)
    input_values = torchaudio.functional.resample(input_values, sr, 16000)

    # encode a waveform into syllabic units
    units = encoder(input_values.to(encoder.device))[0]["units"]  # [3950, 67, ..., 503]
    input_len = len(units)

    # speech language modeling
    messages = [
        {"role": "user", "content": "".join(f"<{unit}>" for unit in units)},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).input_ids.to(speechlm.device)

    generated_ids = speechlm.generate(input_ids=input_ids, do_sample=True, temperature=temperature)[0]

    units = tokenizer.decode(generated_ids)
    units = torch.tensor([int(unit) for unit in re.findall(r"<(\d+)>", units)], device=decoder.device)
    units = units[input_len:]

    # unit-to-speech synthesis
    outputs = decoder(units.unsqueeze(0))
    generated_speech = outputs.waveform.squeeze(0).cpu().numpy()
    boundaries = outputs.durations.squeeze(0).cumsum(0).cpu()

    # Transcript
    generated_text = pipe(generated_speech, generate_kwargs={"language": "english"}, return_timestamps=True)["text"]

    spectrogram = transform(outputs.waveform.squeeze(0))
    spectrogram = torch.log(torch.clamp(spectrogram, min=1e-5))
    spectrogram = spectrogram.cpu().numpy()

    ticks = torch.cat([torch.tensor([0]), boundaries])
    ticks = (ticks[1:] + ticks[:-1]) // 2

    plt.figure(figsize=[25.6, 4.8])
    plt.imshow(spectrogram)
    plt.vlines(boundaries.numpy()[:-1], 0, 79, colors="red")
    plt.xticks(ticks=ticks, labels=[str(unit) for unit in units.tolist()], rotation=270, fontsize=10)
    plt.yticks([], [])
    plt.savefig("spectrogram.png", bbox_inches="tight")

    return (16000, generated_speech), generated_text, "spectrogram.png"


def load_audio(choice):
    torchaudio.save(
        "input.wav",
        dataset[choice]["audio"]["array"].unsqueeze(0),
        dataset[choice]["audio"]["sampling_rate"],
    )
    return "input.wav"


if __name__ == "__main__":
    with gr.Blocks(title="Spoken question answering") as demo:
        with gr.Row():
            choices = [(f"Q{n}: {q}", n) for n, q in enumerate(dataset["question"])]
            dropdown = gr.Dropdown(choices=choices, value=0, type="index", label="Question")
            audio_in = gr.Audio(type="filepath", label="Speech input")

        with gr.Column():
            temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.1, label="Temperature")

        with gr.Column():
            btn = gr.Button("Generate")
            audio_out = gr.Audio(label="Generated speech", streaming=True, autoplay=True)
            text_out = gr.Textbox(label="Transcript")
            plot_out = gr.Image(type="filepath", label="Syllabic tokenization")

        demo.load(fn=load_audio, inputs=dropdown, outputs=audio_in)
        dropdown.change(fn=load_audio, inputs=dropdown, outputs=audio_in)
        btn.click(main, inputs=[audio_in, temperature], outputs=[audio_out, text_out, plot_out])

    demo.launch()
