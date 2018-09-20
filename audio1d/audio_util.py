import librosa


class Audio:
    def __init__(self, hparams):
        self.hparams = hparams

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.hparams.sample_rate)[0]

    def trim(self, wav):
        unused_trimmed, index = librosa.effects.trim(wav)
        start_idx = max(index[0] - self.hparams.num_sil_samples, 0)
        stop_idx = min(index[1] + self.hparams.num_sil_samples, len(wav))
        trimmed = wav[start_idx:stop_idx]
        return trimmed
