# Step-by-Step Guide: Building an AI-Powered Video Translator with Voice Dubbing and Subtitles

## Overview

This guide will walk you through building a free, open-source web application that translates uploaded videos into a chosen language (primarily English) with AI-generated voice dubbing and subtitle generation. The application uses Python with FastAPI for the backend and minimal frontend rendering, leveraging open-source tools throughout.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Setup](#project-setup)
3. [Installing Dependencies](#installing-dependencies)
4. [Setting Up GPU Support](#setting-up-gpu-support)
5. [Testing Individual Components](#testing-individual-components)
6. [Developing Core Functionality](#developing-core-functionality)
   - [Audio Extraction from Video](#audio-extraction-from-video)
   - [Speech-to-Text Transcription](#speech-to-text-transcription)
   - [Language Translation](#language-translation)
   - [Subtitle Generation](#subtitle-generation)
7. [Voice Dubbing with AI Voice Cloning](#voice-dubbing-with-ai-voice-cloning)
   - [Setting Up Coqui TTS](#setting-up-coqui-tts)
   - [Voice Cloning (Optional)](#voice-cloning-optional)
   - [Integrating TTS with Translation](#integrating-tts-with-translation)
8. [Video Integration](#video-integration)
9. [Building the Web Application with FastAPI](#building-the-web-application-with-fastapi)
   - [Setting Up the Backend](#setting-up-the-backend)
   - [Creating the Frontend](#creating-the-frontend)
10. [Deployment and Optimization](#deployment-and-optimization)
11. [Future Enhancements](#future-enhancements)
12. [Conclusion](#conclusion)

---

## Prerequisites

- **Programming Knowledge**: Python, basic web development (FastAPI)
- **Tools Required**:
  - Python 3.7 or later
  - Git
  - NVIDIA GPU with CUDA support (recommended for performance)
- **Libraries and Frameworks**:
  - FFmpeg
  - OpenAI Whisper
  - MarianMT (from Hugging Face Transformers)
  - Coqui TTS
  - FastAPI

---

## Project Setup

### 1. Create a Project Directory

Open your terminal or command prompt and create a new directory for your project:

```bash
mkdir ai_video_translator
cd ai_video_translator
```

### 2. Initialize a Git Repository

Initialize an empty Git repository for version control:

```bash
git init
```

### 3. Set Up a Virtual Environment

It's a good practice to use a virtual environment to manage your project's dependencies:

```bash
python3 -m venv venv
```

Activate the virtual environment:

- **On Unix or MacOS:**

  ```bash
  source venv/bin/activate
  ```

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

---

## Installing Dependencies

### 1. Upgrade Pip

Before installing packages, ensure that `pip` is up-to-date:

```bash
pip install --upgrade pip
```

### 2. Install FFmpeg

FFmpeg is a multimedia framework used for audio and video processing.

- **On Ubuntu/Debian:**

  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```

- **On MacOS (using Homebrew):**

  ```bash
  brew install ffmpeg
  ```

- **On Windows:**

  Download FFmpeg from the [official website](https://ffmpeg.org/download.html) and add the `bin` folder to your system PATH.

### 3. Install Python Libraries

Create a `requirements.txt` file with the following content:

```txt
fastapi
uvicorn[standard]
ffmpeg-python
pydantic
torch
torchaudio
transformers
sentencepiece
TTS
openai-whisper
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Setting Up GPU Support

To leverage GPU acceleration, install PyTorch with CUDA support.

### 1. Install PyTorch with CUDA

Visit the [PyTorch Get Started Page](https://pytorch.org/get-started/locally/) and select the appropriate command based on your system and CUDA version.

For example, if you have CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Verify CUDA Installation

Run the following Python script to verify that PyTorch can access the GPU:

```python
import torch

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
```

---

## Testing Individual Components

Before integrating everything, test each component individually.

### 1. Test FFmpeg for Audio Extraction

Extract audio from a sample video:

```bash
ffmpeg -i input_video.mp4 -q:a 0 -map a output_audio.wav
```

### 2. Test OpenAI Whisper for Transcription

```python
import whisper

model = whisper.load_model("base")  # You can choose "tiny", "small", "medium", "large"
result = model.transcribe("output_audio.wav")
print(result["text"])
```

### 3. Test Translation with MarianMT

```python
from transformers import MarianMTModel, MarianTokenizer

# Load the pre-trained model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-ru-en'  # Example: Russian to English
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Prepare the text
src_text = [ "Hello, how are you?" ]

# Tokenize and translate
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))

# Decode the translations
tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(tgt_text)
```

### 4. Test Coqui TTS for Text-to-Speech

```python
from TTS.api import TTS

# Initialize the TTS model (change model_name to your preference)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)

# Synthesize speech
tts.tts_to_file(text="This is a test sentence.", file_path="tts_output.wav")
```

---

## Developing Core Functionality

Now that individual components are working, start integrating them.

### Project Structure

Create a basic project structure:

```
ai_video_translator/
├── app/
│   ├── main.py
│   ├── dependencies.py
│   ├── routers/
│   │   ├── __init__.py
│   │   └── translate.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_extraction.py
│   │   ├── speech_to_text.py
│   │   ├── translation.py
│   │   ├── text_to_speech.py
│   │   └── video_processing.py
│   └── models/
│       └── __init__.py
├── data/
├── requirements.txt
└── README.md
```

### 1. Audio Extraction from Video

Create `audio_extraction.py` in `services/` directory.

```python
# app/services/audio_extraction.py

import ffmpeg
import os

def extract_audio(input_video_path, output_audio_path):
    stream = ffmpeg.input(input_video_path)
    audio = stream.audio
    ffmpeg.output(audio, output_audio_path).overwrite_output().run()
```

Usage example:

```python
extract_audio('input_video.mp4', 'output_audio.wav')
```

### 2. Speech-to-Text Transcription

Create `speech_to_text.py` in `services/` directory.

```python
# app/services/speech_to_text.py

import whisper

def transcribe_audio(audio_path, model_size='base'):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result  # Contains 'text' and 'segments' with timestamps
```

Usage example:

```python
transcription_result = transcribe_audio('output_audio.wav')
print(transcription_result['text'])
```

### 3. Language Translation

Create `translation.py` in `services/` directory.

```python
# app/services/translation.py

from transformers import MarianMTModel, MarianTokenizer

def load_translation_model(src_lang='en', tgt_lang='fr'):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text_list, model, tokenizer):
    translated = model.generate(**tokenizer(text_list, return_tensors="pt", padding=True))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return tgt_text
```

Usage example:

```python
model, tokenizer = load_translation_model(src_lang='en', tgt_lang='fr')
translated_text = translate_text([transcription_result['text']], model, tokenizer)
print(translated_text)
```

### 4. Subtitle Generation

Create `subtitle_generation.py` in `services/` directory.

```python
# app/services/subtitle_generation.py

import srt
from datetime import timedelta

def create_srt(transcription_segments, translated_segments, srt_file_path):
    subtitles = []
    for i, segment in enumerate(transcription_segments):
        start = timedelta(seconds=segment['start'])
        end = timedelta(seconds=segment['end'])
        content = translated_segments[i]
        subtitle = srt.Subtitle(index=i+1, start=start, end=end, content=content)
        subtitles.append(subtitle)

    srt_data = srt.compose(subtitles)

    with open(srt_file_path, 'w', encoding='utf-8') as f:
        f.write(srt_data)
```

Usage example:

```python
from services.subtitle_generation import create_srt

transcription_segments = transcription_result['segments']
translated_segments = translate_text([seg['text'] for seg in transcription_segments], model, tokenizer)
create_srt(transcription_segments, translated_segments, 'output_subtitles.srt')
```

---

## Voice Dubbing with AI Voice Cloning

### 1. Setting Up Coqui TTS

We will use Coqui TTS for text-to-speech synthesis.

```python
# app/services/text_to_speech.py

from TTS.api import TTS

def synthesize_speech(text_list, output_audio_path, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
    tts = TTS(model_name=model_name, progress_bar=False, gpu=True)
    concatenated_text = " ".join(text_list)
    tts.tts_to_file(text=concatenated_text, file_path=output_audio_path)
```

Usage example:

```python
synthesize_speech(translated_segments, 'dubbed_audio.wav')
```

### 2. Voice Cloning (Optional)

Voice cloning requires training a TTS model on the desired voice data, which can be complex and time-consuming. For this guide, we'll use a pre-trained voice.

### 3. Integrating TTS with Translation

Ensure that the TTS output matches the timing of the original speech.

```python
def synthesize_speech_segments(translated_segments, transcription_segments, output_dir, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
    tts = TTS(model_name=model_name, progress_bar=False, gpu=True)
    os.makedirs(output_dir, exist_ok=True)
    audio_segments = []

    for i, text in enumerate(translated_segments):
        audio_path = os.path.join(output_dir, f'segment_{i}.wav')
        tts.tts_to_file(text=text, file_path=audio_path)
        audio_segments.append((audio_path, transcription_segments[i]['start'], transcription_segments[i]['end']))

    return audio_segments
```

Usage example:

```python
audio_segments = synthesize_speech_segments(translated_segments, transcription_segments, 'tts_segments')
```

---

## Video Integration

### 1. Combine Dubbed Audio and Subtitles with Original Video

Create `video_processing.py` in `services/` directory.

```python
# app/services/video_processing.py

import ffmpeg

def merge_audio_segments(audio_segments, output_audio_path):
    inputs = [ffmpeg.input(path) for path, _, _ in audio_segments]
    joined = ffmpeg.concat(*inputs, v=0, a=1).output(output_audio_path).overwrite_output()
    joined.run()

def merge_audio_video(original_video_path, dubbed_audio_path, subtitles_path, output_video_path):
    input_video = ffmpeg.input(original_video_path)
    input_audio = ffmpeg.input(dubbed_audio_path)
    input_subs = ffmpeg.input(subtitles_path)

    (
        ffmpeg
        .concat(input_video.video.filter('subtitles', subtitles_path), input_audio.audio, v=1, a=1)
        .output(output_video_path)
        .overwrite_output()
        .run()
    )
```

Usage example:

```python
merge_audio_segments(audio_segments, 'dubbed_audio_full.wav')
merge_audio_video('input_video.mp4', 'dubbed_audio_full.wav', 'output_subtitles.srt', 'final_output_video.mp4')
```

---

## Building the Web Application with FastAPI

### Directory Structure Update

```
ai_video_translator/
├── app/
│   ├── main.py
│   ├── routers/
│   │   ├── __init__.py
│   │   └── translate.py
│   ├── services/
│   ├── models/
│   └── templates/
│       └── index.html
├── static/
│   └── css/
├── data/
├── requirements.txt
└── README.md
```

### 1. Setting Up the Backend

Create `main.py` in the `app/` directory.

```python
# app/main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from app.routers import translate

app = FastAPI()

app.include_router(translate.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI-Powered Video Translator!"}
```

Create `translate.py` in `app/routers/`.

```python
# app/routers/translate.py

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
import os
from app.services import (
    audio_extraction,
    speech_to_text,
    translation,
    subtitle_generation,
    text_to_speech,
    video_processing,
)

router = APIRouter()

@router.post("/translate/")
async def translate_video(language: str = Form(...), file: UploadFile = File(...)):
    # Save uploaded video
    video_path = f"data/{file.filename}"
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    # Processing paths
    audio_path = f"data/audio.wav"
    transcription_result = speech_to_text.transcribe_audio(audio_path)
    segments = transcription_result['segments']

    # Load translation model
    model, tokenizer = translation.load_translation_model(src_lang='auto', tgt_lang=language)

    # Translate segments
    translated_segments = translation.translate_text([seg['text'] for seg in segments], model, tokenizer)

    # Generate subtitles
    srt_path = f"data/subtitles.srt"
    subtitle_generation.create_srt(segments, translated_segments, srt_path)

    # Synthesize speech
    audio_segments = text_to_speech.synthesize_speech_segments(translated_segments, segments, 'data/tts_segments')

    # Merge audio segments
    dubbed_audio_path = f"data/dubbed_audio.wav"
    video_processing.merge_audio_segments(audio_segments, dubbed_audio_path)

    # Merge audio, video, subtitles
    output_video_path = f"data/translated_{file.filename}"
    video_processing.merge_audio_video(video_path, dubbed_audio_path, srt_path, output_video_path)

    return FileResponse(output_video_path, media_type="video/mp4", filename=f"translated_{file.filename}")
```

### 2. Creating the Frontend

Create an `index.html` template in `app/templates/`.

```html
<!-- app/templates/index.html -->

<!DOCTYPE html>
<html>
<head>
    <title>AI-Powered Video Translator</title>
</head>
<body>
    <h1>Upload Video for Translation</h1>
    <form action="/translate/" method="post" enctype="multipart/form-data">
        <label for="language">Target Language:</label>
        <input type="text" id="language" name="language" placeholder="en, es, fr, etc."><br><br>
        <label for="file">Select video:</label>
        <input type="file" id="file" name="file"><br><br>
        <input type="submit" value="Translate Video">
    </form>
</body>
</html>
```

Modify `main.py` to render the template:

```python
# app/main.py

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

# ... existing imports ...

templates = Jinja2Templates(directory="app/templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
```

---

## Deployment and Optimization

### 1. Running the Application Locally

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --reload
```

### 2. Deployment Options

- **Local Server with GPU**: Run the application on a machine with a GPU.
- **Cloud Platforms**: Deploy to cloud services that provide GPU support (e.g., AWS EC2 with GPU, Google Cloud, or Azure).
- **Dockerization**: Containerize the application using Docker for consistent deployment environments.

### 3. Optimization Tips

- **Model Loading**: Load models once at startup to avoid reloading them on every request.
- **Batch Processing**: Implement batch processing if handling multiple requests.
- **Error Handling**: Add comprehensive error handling and validation.
- **Security**: Implement secure file handling and input validation to prevent vulnerabilities.

---

## Future Enhancements

- **Expanded Language Support**: Incorporate more language models as they become available.
- **Advanced Voice Cloning**: Experiment with models like **YourTTS** for multilingual voice cloning.
- **User Accounts**: Implement user authentication to save and manage translated videos.
- **Real-time Translation**: Investigate streaming solutions for real-time or near-real-time translation.

---

## Conclusion

You have now built a functional AI-powered video translator that accepts video uploads, translates the audio into a target language with dubbed voice and subtitles, and allows users to download the translated video. This application uses open-source tools and follows best practices to ensure a maintainable and scalable codebase.

---

**Note**: This guide provides a high-level overview and code snippets to get you started. Be sure to handle exceptions, edge cases, and input validation in your actual implementation. Additionally, always respect intellectual property laws and ensure you have the right to process and redistribute any media through your application.

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI Whisper Repository](https://github.com/openai/whisper)
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
- [Coqui TTS Documentation](https://tts.readthedocs.io/en/latest/)
- [FFmpeg Documentation](https://ffmpeg.org/ffmpeg.html)