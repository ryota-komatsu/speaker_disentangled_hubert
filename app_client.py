import io
import json

import gradio as gr
import soundfile as sf
import websockets
from datasets import Audio, load_dataset
from PIL import Image

SERVER_URL = "ws://127.0.0.1:8000"

dataset = load_dataset("fixie-ai/llama-questions", split="test")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))


def load_audio(choice):
    return dataset[choice]["audio"]["sampling_rate"], dataset[choice]["audio"]["array"]


async def main(audio_path, temperature):
    async with websockets.connect(SERVER_URL, max_size=None) as websocket:
        with open(audio_path, "rb") as f:
            input_audio = f.read()

        await websocket.send(json.dumps({"temperature": temperature}))
        await websocket.send(input_audio)

        text = await websocket.recv()
        output_audio = await websocket.recv()
        spectrogram = await websocket.recv()

        output_audio, sr = sf.read(io.BytesIO(output_audio))
        spectrogram = Image.open(io.BytesIO(spectrogram))

        return (sr, output_audio), text, spectrogram


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
        plot_out = gr.Image(label="Syllabic tokenization")

    demo.load(fn=load_audio, inputs=dropdown, outputs=audio_in)
    dropdown.change(fn=load_audio, inputs=dropdown, outputs=audio_in)
    btn.click(main, inputs=[audio_in, temperature], outputs=[audio_out, text_out, plot_out])


if __name__ == "__main__":
    demo.launch()
