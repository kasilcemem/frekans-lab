import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa


SAMPLE_RATE = 44100
CHANNELS = 1


def mikrofon_kaydet(sure: float = 3.0, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Mikrofondan ses kaydeder.
    sure: kayıt süresi (saniye)
    döner: (ses_verisi, sample_rate)
    """
    print(f"Kayıt başlıyor... {sure} saniye")
    veri = sd.rec(
        int(sure * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32"
    )
    sd.wait()
    print("Kayıt tamamlandı.")
    return veri.flatten(), sample_rate


def wav_oku(dosya_yolu: str) -> tuple[np.ndarray, int]:
    """
    .wav dosyasını okur.
    döner: (ses_verisi, sample_rate)
    """
    sample_rate, veri = wav.read(dosya_yolu)
    if veri.ndim > 1:
        veri = veri.mean(axis=1)
    veri = veri.astype(np.float32)
    if veri.max() > 1.0:
        veri = veri / 32768.0
    return veri, sample_rate


def mp3_oku(dosya_yolu: str) -> tuple[np.ndarray, int]:
    """
    .mp3 dosyasını okur (librosa ile).
    döner: (ses_verisi, sample_rate)
    """
    veri, sample_rate = librosa.load(dosya_yolu, sr=None, mono=True)
    return veri, sample_rate


def dosya_oku(dosya_yolu: str) -> tuple[np.ndarray, int]:
    """
    Uzantıya göre otomatik dosya okuyucu.
    .wav ve .mp3 destekler.
    """
    if dosya_yolu.endswith(".wav"):
        return wav_oku(dosya_yolu)
    elif dosya_yolu.endswith(".mp3"):
        return mp3_oku(dosya_yolu)
    else:
        raise ValueError(f"Desteklenmeyen format: {dosya_yolu}")


if __name__ == "__main__":
    veri, sr = mikrofon_kaydet(sure=3.0)
    print(f"Örnek sayısı: {len(veri)}, Sample rate: {sr}")
