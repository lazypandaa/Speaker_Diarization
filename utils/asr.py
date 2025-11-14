import whisper

model = whisper.load_model("tiny")   # tiny = fastest for CPU

def transcribe_audio(path):
    print("Transcribing with Whisper...")
    return model.transcribe(path)
