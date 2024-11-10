import logging
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/translate/")
async def translate_video(source_language: str = Form(...), language: str = Form(...), file: UploadFile = File(...)):
    try:
        # Ensure the data directory exists
        os.makedirs("data", exist_ok=True)

        # Save uploaded video
        video_path = f"data/{file.filename}"
        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"Video saved to {video_path}")

        # Extract audio for transcription
        audio_path = f"data/audio.wav"
        audio_extraction.extract_audio(video_path, audio_path)
        logger.info(f"Audio extracted for transcription to {audio_path}")

        # Transcribe audio to text
        transcription_result = speech_to_text.transcribe_audio(audio_path)
        segments = transcription_result['segments']
        logger.info("Audio transcribed to text")

        # Load translation model
        model, tokenizer = translation.load_translation_model(src_lang=source_language, tgt_lang=language)
        logger.info("Translation model loaded")

        # Translate segments
        translated_segments = translation.translate_text([seg['text'] for seg in segments], model, tokenizer)
        logger.info("Text translated")

        # Generate subtitles
        srt_path = f"data/subtitles.srt"
        subtitle_generation.create_srt(segments, translated_segments, srt_path)
        logger.info(f"Subtitles generated at {srt_path}")

        # Synthesize speech
        audio_segments = text_to_speech.synthesize_speech_segments(translated_segments, segments, 'data/tts_segments')
        logger.info("Speech synthesized")

        # Merge audio segments
        dubbed_audio_path = f"data/dubbed_audio.wav"
        video_processing.merge_audio_segments(audio_segments, dubbed_audio_path)
        logger.info(f"Dubbed audio saved to {dubbed_audio_path}")

        # Merge audio, video, and subtitles
        output_video_path = f"data/translated_{file.filename}"
        video_processing.merge_audio_video(video_path, dubbed_audio_path, srt_path, output_video_path)
        logger.info(f"Final video saved to {output_video_path}")

        return FileResponse(output_video_path, media_type="video/mp4", filename=f"translated_{file.filename}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise