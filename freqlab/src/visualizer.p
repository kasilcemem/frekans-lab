import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def spektrum_ciz(frekanslar: np.ndarray, genlikler: np.ndarray, baslik: str = "Frekans Spektrumu"):
    """
    FFT spektrum grafiği çizer.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frekanslar, genlikler, color="#378ADD", linewidth=0.8)
    ax.set_xlabel("Frekans (Hz)")
    ax.set_ylabel("Genlik")
    ax.set_title(baslik)
    ax.set_xlim(0, min(frekanslar.max(), 20000))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def db_spektrum_ciz(frekanslar: np.ndarray, db_deger: np.ndarray):
    """
    dB cinsinden frekans spektrumu çizer.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogx(frekanslar[1:], db_deger[1:], color="#1D9E75", linewidth=0.8)
    ax.set_xlabel("Frekans (Hz) — log skala")
    ax.set_ylabel("Genlik (dB)")
    ax.set_title("Frekans Spektrumu (dB)")
    ax.set_xlim(20, 20000)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


def dalga_formu_ciz(veri: np.ndarray, sample_rate: int):
    """
    Zaman domeninde dalga formu çizer.
    """
    sure = len(veri) / sample_rate
    zaman = np.linspace(0, sure, len(veri))
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(zaman, veri, color="#D85A30", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Zaman (sn)")
    ax.set_ylabel("Genlik")
    ax.set_title("Dalga Formu")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def spektrogram_ciz(veri: np.ndarray, sample_rate: int):
    """
    Mel spektrogramı çizer (zaman x frekans x güç).
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    S = librosa.feature.melspectrogram(y=veri, sr=sample_rate, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_db, sr=sample_rate, x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spektrogramı")
    plt.tight_layout()
    plt.show()


def bant_grafigi_ciz(bantlar: dict):
    """
    Frekans bantlarının güç dağılımını bar grafik olarak gösterir.
    """
    renkler = ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bantlar.keys(), bantlar.values(), color=renkler, edgecolor="none")
    ax.set_xlabel("Frekans Bandı")
    ax.set_ylabel("Enerji")
    ax.set_title("Bant Güç Dağılımı")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def tum_grafikleri_goster(sonuc: dict):
    """
    Tüm grafikleri tek seferde çizer.
    """
    dalga_formu_ciz(sonuc["veri"], sonuc["sample_rate"])
    spektrum_ciz(sonuc["frekanslar"], sonuc["genlikler"])
    db_spektrum_ciz(sonuc["frekanslar"], sonuc["db"])
    spektrogram_ciz(sonuc["veri"], sonuc["sample_rate"])
    bant_grafigi_ciz(sonuc["bantlar"])
