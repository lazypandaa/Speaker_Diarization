import webrtcvad
import librosa
import numpy as np

def get_speech_segments(path, frame_ms=30):
    audio, sr = librosa.load(path, sr=16000)
    audio_i16 = (audio * 32768).astype(np.int16)

    vad = webrtcvad.Vad(2)
    frame_len = int(sr * frame_ms / 1000)

    raw_segments = []
    start = None

    for i in range(0, len(audio_i16), frame_len):
        frame = audio_i16[i:i+frame_len].tobytes()
        if len(frame) < frame_len * 2:
            break

        is_speech = vad.is_speech(frame, sr)

        if is_speech and start is None:
            start = i
        elif not is_speech and start is not None:
            raw_segments.append((start, i))
            start = None

    if start is not None:
        raw_segments.append((start, len(audio_i16)))

    # ---------- MERGE SMALL SEGMENTS ----------
    merged = []
    min_gap = sr * 0.5  # merge if silence < 0.5 sec

    cur_start, cur_end = raw_segments[0]

    for s, e in raw_segments[1:]:
        if s - cur_end <= min_gap:
            cur_end = e  # merge
        else:
            merged.append({"start": cur_start, "end": cur_end})
            cur_start, cur_end = s, e

    merged.append({"start": cur_start, "end": cur_end})

    return merged, audio, sr
