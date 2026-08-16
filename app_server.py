import asyncio
import io
import re
from contextlib import asynccontextmanager

import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, pipeline

from src.flow_matching import FlowMatchingWithBigVGan
from src.s5hubert import SylRegForSyllableDiscovery


class Model:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # download pretrained models from hugging face hub
        self.encoder = SylRegForSyllableDiscovery.from_pretrained("ryota-komatsu/SylReg-Distill", device_map=device)
        self.decoder = FlowMatchingWithBigVGan.from_pretrained("ryota-komatsu/SylReg-Decoder", device_map=device)
        self.speechlm = AutoModelForCausalLM.from_pretrained(
            "ryota-komatsu/SylReg-LM-7B-Instruct", device_map=device, dtype="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("ryota-komatsu/SylReg-LM-7B-Instruct")

        asr = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3",
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=asr,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
        )

        self.transform = torchaudio.transforms.MelSpectrogram(hop_length=320, n_mels=80, center=False).to(device)

    @torch.inference_mode()
    def __call__(self, input_audio: bytes, temperature: float):
        # load a waveform
        input_values = io.BytesIO(input_audio)
        input_values, sr = torchaudio.load(input_values)
        input_values = torchaudio.functional.resample(input_values, sr, 16000)

        # encode a waveform into syllabic units
        units = self.encoder(input_values.to(self.encoder.device))[0]["units"]  # [3950, 67, ..., 503]
        input_len = len(units)

        # speech language modeling
        messages = [
            {"role": "user", "content": "".join(f"<{unit}>" for unit in units)},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).input_ids.to(self.speechlm.device)

        generated_ids = self.speechlm.generate(input_ids=input_ids, do_sample=True, temperature=temperature)[0]

        units = self.tokenizer.decode(generated_ids)
        units = torch.tensor([int(unit) for unit in re.findall(r"<(\d+)>", units)], device=self.decoder.device)
        units = units[input_len:]

        # unit-to-speech synthesis
        outputs = self.decoder(units.unsqueeze(0))
        generated_speech = outputs.waveform.squeeze(0).cpu().numpy()
        boundaries = outputs.durations.squeeze(0).cumsum(0).cpu()

        # Transcript
        generated_text = self.pipe(generated_speech, generate_kwargs={"language": "english"}, return_timestamps=True)[
            "text"
        ]

        spectrogram = self.transform(outputs.waveform.squeeze(0))
        spectrogram = torch.log(torch.clamp(spectrogram, min=1e-5))
        spectrogram = spectrogram.cpu().numpy()

        ticks = torch.cat([torch.tensor([0]), boundaries])
        ticks = (ticks[1:] + ticks[:-1]) // 2

        audio_buffer = io.BytesIO()
        spectrogram_buffer = io.BytesIO()

        sf.write(audio_buffer, generated_speech, 16000, format="WAV")

        plt.figure(figsize=[25.6, 4.8])
        plt.imshow(spectrogram)
        plt.vlines(boundaries.numpy()[:-1], 0, 79, colors="red")
        plt.xticks(ticks=ticks, labels=[str(unit) for unit in units.tolist()], rotation=270, fontsize=10)
        plt.yticks([], [])
        plt.savefig(spectrogram_buffer, format="png", bbox_inches="tight")

        return audio_buffer.getvalue(), generated_text, spectrogram_buffer.getvalue()


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["model"] = Model()
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            config = await websocket.receive_json()
            input_audio = await websocket.receive_bytes()

            output_audio, text, spectrogram = await asyncio.to_thread(
                ml_models["model"], input_audio, config["temperature"]
            )

            await websocket.send_text(text)
            await websocket.send_bytes(output_audio)
            await websocket.send_bytes(spectrogram)

    except WebSocketDisconnect:
        pass
