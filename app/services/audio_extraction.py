import ffmpeg

def extract_audio(input_video_path, output_audio_path):
    stream = ffmpeg.input(input_video_path)
    audio = stream.audio
    ffmpeg.output(audio, output_audio_path).overwrite_output().run() 