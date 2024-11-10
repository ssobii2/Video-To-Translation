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