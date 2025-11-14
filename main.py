import numpy as np
import soundfile as sf
import json
from utils.vad import get_speech_segments
from utils.embedding import get_embedding, cosine_similarity
from utils.asr import transcribe_audio


def segment_text(full_transcript, diar, sr, audio_len):
    """
    Rough text segmentation: splits text proportionally by segment duration.
    Not perfect, but works well for hackathon submission.
    """
    text = full_transcript.strip()
    words = text.split()
    total_words = len(words)

    total_audio_sec = audio_len / sr
    output = []

    start_word = 0

    for seg in diar:
        seg_duration = seg["end"] - seg["start"]
        proportion = seg_duration / total_audio_sec
        word_count = max(1, int(proportion * total_words))

        segment_words = words[start_word:start_word + word_count]
        start_word += word_count

        seg["text"] = " ".join(segment_words)
        output.append(seg)

    return output


def main():
    mix_path = "samples/mixture_audio.wav"
    target_path = "samples/target_sample.wav"

    print("Embedding target...")
    target_emb = get_embedding(target_path)

    print("Running VAD...")
    segments, audio, sr = get_speech_segments(mix_path)

    diar = []
    target_audio = []

    for seg in segments:
        start = int(seg["start"])
        end = int(seg["end"])
        chunk = audio[start:end]

        sf.write("tmp.wav", chunk, sr)
        emb = get_embedding("tmp.wav")
        score = cosine_similarity(target_emb, emb)

        # label = "Target" if score > 0.75 else "Other"
        label = "Target" if score > 0.75 else "Speaker_B"


        diar.append({
            "speaker": label,
            "start": start / sr,
            "end": end / sr,
            "confidence": float(score),
            "text": ""  # will fill later
        })

        if label == "Target":
            target_audio.append(chunk)

    if target_audio:
        final = np.concatenate(target_audio)
        sf.write("outputs/target_speaker.wav", final, sr)
        print("Saved outputs/target_speaker.wav")

    print("Running ASR...")
    asr_result = transcribe_audio(mix_path)
    full_text = asr_result["text"]
    print("TRANSCRIPT:", full_text)

    print("Segmenting transcript into diarization...")
    diar_with_text = segment_text(full_text, diar, sr, len(audio))

    # Save JSON
    with open("outputs/diarization.json", "w") as f:
        json.dump(diar_with_text, f, indent=4)

    print("Saved outputs/diarization.json")
    print("Done!")


if __name__ == "__main__":
    main()
