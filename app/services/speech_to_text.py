import whisper
import logging
import torch

logger = logging.getLogger(__name__)

def transcribe_audio(audio_path, model_size='base'):
    logger.info(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        logger.info("Whisper model moved to GPU")
    else:
        logger.info("Using CPU for Whisper model")

    logger.info("Model loaded successfully")

    logger.info(f"Transcribing audio from {audio_path}")
    result = model.transcribe(audio_path, verbose=False)
    logger.info("Transcription complete")
    
    logger.info(f"Transcription result: {result['text']}")
    
    return result  # Contains 'text' and 'segments' with timestamps 