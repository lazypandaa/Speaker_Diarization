from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np

encoder = VoiceEncoder()

def get_embedding(path):
    # preprocess_wav loads + normalizes audio correctly
    wav = preprocess_wav(path)
    emb = encoder.embed_utterance(wav)
    return emb

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
