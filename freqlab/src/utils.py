import numpy as np
from scipy.signal import butter, sosfilt


def nota_bul(frekans: float) -> str:
    """
    Frekansı en yakın müzik notasına çevirir.
    Örnek: 440.0 Hz → A4
    """
    notalar = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]
    if frekans <= 0:
        return "?"
    yarim_ton = 12 * np.log2(frekans / 440.0) + 69
    midi = int(round(yarim_ton))
    oktav = (midi // 12) - 1
    nota = notalar[midi % 12]
    return f"{nota}{oktav}"


def hz_to_mel(hz: float) -> float:
    """Hz → Mel dönüşümü."""
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel: float) -> float:
    """Mel → Hz dönüşümü."""
    return 700 * (10 ** (mel / 2595) - 1)


def normalize(veri: np.ndarray) -> np.ndarray:
    """Ses verisini -1 ile 1 arasına normalize eder."""
    maks = np.max(np.abs(veri))
    if maks == 0:
        return veri
    return veri / maks


def alçak_geciren_filtre(
    veri: np.ndarray,
    kesim_hz: float,
    sample_rate: int,
    derece: int = 4
) -> np.ndarray:
    """
    Alçak geçiren (low-pass) Butterworth filtresi.
    kesim_hz altındaki frekansları geçirir.
    """
    nyquist = sample_rate / 2
    sos = butter(derece, kesim_hz / nyquist, btype="low", output="sos")
    return sosfilt(sos, veri)


def yüksek_geciren_filtre(
    veri: np.ndarray,
    kesim_hz: float,
    sample_rate: int,
    derece: int = 4
) -> np.ndarray:
    """
    Yüksek geçiren (high-pass) Butterworth filtresi.
    kesim_hz üzerindeki frekansları geçirir.
    """
    nyquist = sample_rate / 2
    sos = butter(derece, kesim_hz / nyquist, btype="high", output="sos")
    return sosfilt(sos, veri)


def bant_geciren_filtre(
    veri: np.ndarray,
    dusuk_hz: float,
    yuksek_hz: float,
    sample_rate: int,
    derece: int = 4
) -> np.ndarray:
    """
    Bant geçiren (band-pass) filtresi.
    dusuk_hz ile yuksek_hz arasındaki frekansları geçirir.
    """
    nyquist = sample_rate / 2
    sos = butter(
        derece,
        [dusuk_hz / nyquist, yuksek_hz / nyquist],
        btype="band",
        output="sos"
    )
    return sosfilt(sos, veri)


def sure_formatla(saniye: float) -> str:
    """Saniyeyi okunabilir formata çevirir. Örnek: 125.3 → '2:05.3'"""
    dakika = int(saniye // 60)
    kalan = saniye % 60
    return f"{dakika}:{kalan:04.1f}"
