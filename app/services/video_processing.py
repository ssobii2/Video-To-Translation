import ffmpeg
import logging

logger = logging.getLogger(__name__)

def merge_audio_segments(audio_segments, output_audio_path):
    logger.info("Merging audio segments...")
    inputs = [ffmpeg.input(path) for path, _, _ in audio_segments]
    joined = ffmpeg.concat(*inputs, v=0, a=1).output(output_audio_path).overwrite_output()

    # Run FFmpeg and capture output
    process = joined.run_async(pipe_stderr=True)

    # Read output and errors in real-time
    while True:
        output = process.stderr.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            logger.info(output.decode().strip())

    logger.info(f"Merged audio segments saved to {output_audio_path}")

def merge_audio_video(original_video_path, dubbed_audio_path, subtitles_path, output_video_path):
    logger.info("Merging video with dubbed audio and subtitles...")
    input_video = ffmpeg.input(original_video_path)
    input_audio = ffmpeg.input(dubbed_audio_path)
    input_subs = ffmpeg.input(subtitles_path)
    
    # Combine video and dubbed audio
    output = (
        ffmpeg
        .output(
            input_video.video.filter('subtitles', subtitles_path), 
            input_audio.audio,
            output_video_path,
            vcodec='h264_nvenc',  # Use NVIDIA encoder for GPU acceleration
            acodec='aac',         # Use AAC codec for audio
            preset='fast'         # Additional encoding options
        )
        .overwrite_output()
    )
    
    # Run FFmpeg and capture output
    process = output.run_async(pipe_stderr=True)

    # Read output and errors in real-time
    while True:
        output = process.stderr.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            logger.info(output.decode().strip())

    logger.info(f"Merged video saved to {output_video_path}")