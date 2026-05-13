import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


def fft_hesapla(veri: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Ses verisine FFT uygular.
    döner: (frekanslar, genlikler)
    """
    n = len(veri)
    frekanslar = fftfreq(n, d=1.0 / sample_rate)
    spektrum = fft(veri)
    genlikler = np.abs(spektrum) * 2 / n

    # Sadece pozitif frekanslar
    pozitif = frekanslar >= 0
    return frekanslar[pozitif], genlikler[pozitif]


def dominant_frekanslar(
    frekanslar: np.ndarray,
    genlikler: np.ndarray,
    top_n: int = 5
) -> list[dict]:
    """
    En güçlü N frekansı döner.
    döner: [{"frekans": Hz, "genlik": deger}, ...]
    """
    peaks, _ = find_peaks(genlikler, height=genlikler.max() * 0.1)
    sirali = peaks[np.argsort(genlikler[peaks])[::-1]]
    sonuc = []
    for i in sirali[:top_n]:
        sonuc.append({
            "frekans": round(float(frekanslar[i]), 2),
            "genlik": round(float(genlikler[i]), 6)
        })
    return sonuc


def bant_guc(
    frekanslar: np.ndarray,
    genlikler: np.ndarray
) -> dict:
    """
    Frekans bantlarına göre güç dağılımını hesaplar.
    döner: {"bass": ..., "mid": ..., "treble": ..., "yuksek": ...}
    """
    def bant_enerjisi(f_min, f_max):
        maske = (frekanslar >= f_min) & (frekanslar < f_max)
        return float(np.sum(genlikler[maske] ** 2))

    return {
        "bass":   round(bant_enerjisi(20, 250), 6),
        "mid":    round(bant_enerjisi(250, 2000), 6),
        "treble": round(bant_enerjisi(2000, 8000), 6),
        "yuksek": round(bant_enerjisi(8000, 20000), 6),
    }


def db_donustur(genlikler: np.ndarray, ref: float = 1.0) -> np.ndarray:
    """
    Genlikleri desibele (dB) çevirir.
    """
    genlikler = np.clip(genlikler, 1e-10, None)
    return 20 * np.log10(genlikler / ref)


def analiz_et(veri: np.ndarray, sample_rate: int) -> dict:
    """
    Tüm analizi tek fonksiyonda çalıştırır.
    döner: tam analiz sonucu dict
    """
    frekanslar, genlikler = fft_hesapla(veri, sample_rate)
    return {
        "frekanslar": frekanslar,
        "genlikler": genlikler,
        "dominant": dominant_frekanslar(frekanslar, genlikler),
        "bantlar": bant_guc(frekanslar, genlikler),
        "db": db_donustur(genlikler),
        "sample_rate": sample_rate,
        "sure": round(len(veri) / sample_rate, 3),
    }


if __name__ == "__main__":
    from audio_input import dosya_oku
    import sys

    dosya = sys.argv[1] if len(sys.argv) > 1 else "data/samples/ornek.wav"
    veri, sr = dosya_oku(dosya)
    sonuc = analiz_et(veri, sr)

    print(f"Süre: {sonuc['sure']} sn | Sample rate: {sr} Hz")
    print("Dominant frekanslar:")
    for d in sonuc["dominant"]:
        print(f"  {d['frekans']} Hz — genlik: {d['genlik']}")
    print("Bant güçleri:", sonuc["bantlar"])
