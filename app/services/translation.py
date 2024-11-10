from transformers import MarianMTModel, MarianTokenizer
import logging
import torch

logger = logging.getLogger(__name__)

def load_translation_model(src_lang='en', tgt_lang='fr'):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    logger.info(f"Loading translation model: {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        logger.info("Translation model moved to GPU")
    else:
        logger.info("Using CPU for translation model")

    logger.info("Translation model loaded successfully")
    return model, tokenizer

def translate_text(text_list, model, tokenizer, batch_size=4):
    logger.info("Starting translation process")
    translated_texts = []

    # Prepare input tensors and move them to the same device as the model
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        logger.info(f"Translating batch {i // batch_size + 1}: {batch}")

        input_tensors = tokenizer(batch, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            input_tensors = {key: value.to('cuda') for key, value in input_tensors.items()}

        translated = model.generate(**input_tensors)
        for j in range(len(batch)):
            tgt_text = tokenizer.decode(translated[j], skip_special_tokens=True)
            translated_texts.append(tgt_text)
            logger.info(f"Segment {i + j + 1} translated: {tgt_text}")

    logger.info("Translation process completed")
    return translated_texts 