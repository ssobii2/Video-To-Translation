from TTS.api import TTS
import os
import logging

logger = logging.getLogger(__name__)

def synthesize_speech_segments(translated_segments, transcription_segments, output_dir, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
    tts = TTS(model_name=model_name, progress_bar=False, gpu=True)
    os.makedirs(output_dir, exist_ok=True)
    audio_segments = []

    for i, text in enumerate(translated_segments):
        audio_path = os.path.join(output_dir, f'segment_{i}.wav')
        logger.info(f"Synthesizing speech for segment {i + 1}/{len(translated_segments)}: {text}")
        tts.tts_to_file(text=text, file_path=audio_path)
        logger.info(f"Synthesized audio saved to {audio_path}")
        audio_segments.append((audio_path, transcription_segments[i]['start'], transcription_segments[i]['end']))

    return audio_segments 