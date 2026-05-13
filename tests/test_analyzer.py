import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analyzer import (
    fft_hesapla,
    dominant_frekanslar,
    bant_guc,
    db_donustur,
    analiz_et,
)
from utils import (
    nota_bul,
    normalize,
    hz_to_mel,
    mel_to_hz,
    alçak_geciren_filtre,
    yüksek_geciren_filtre,
    bant_geciren_filtre,
)


SAMPLE_RATE = 44100
SURE = 1.0


def saf_ses_uret(frekans: float, sample_rate: int = SAMPLE_RATE, sure: float = SURE) -> np.ndarray:
    """Test için belirli frekansta saf sinüs sinyali üretir."""
    t = np.linspace(0, sure, int(sample_rate * sure), endpoint=False)
    return np.sin(2 * np.pi * frekans * t).astype(np.float32)


# ── FFT testleri ──────────────────────────────────────────────────────────────

class TestFFT:

    def test_frekans_uzunlugu(self):
        veri = saf_ses_uret(440)
        frekanslar, genlikler = fft_hesapla(veri, SAMPLE_RATE)
        assert len(frekanslar) == len(genlikler)

    def test_pozitif_frekanslar(self):
        veri = saf_ses_uret(440)
        frekanslar, _ = fft_hesapla(veri, SAMPLE_RATE)
        assert np.all(frekanslar >= 0)

    def test_dogru_frekans_tespiti(self):
        hedef = 440.0
        veri = saf_ses_uret(hedef)
        frekanslar, genlikler = fft_hesapla(veri, SAMPLE_RATE)
        en_guclu = frekanslar[np.argmax(genlikler)]
        assert abs(en_guclu - hedef) < 2.0, f"Beklenen ~{hedef} Hz, bulunan {en_guclu} Hz"

    def test_sessizlik(self):
        veri = np.zeros(SAMPLE_RATE, dtype=np.float32)
        frekanslar, genlikler = fft_hesapla(veri, SAMPLE_RATE)
        assert genlikler.max() < 1e-6


# ── Dominant frekans testleri ─────────────────────────────────────────────────

class TestDominantFrekanslar:

    def test_sonuc_tipi(self):
        veri = saf_ses_uret(1000)
        f, g = fft_hesapla(veri, SAMPLE_RATE)
        sonuc = dominant_frekanslar(f, g, top_n=3)
        assert isinstance(sonuc, list)
        assert len(sonuc) <= 3

    def test_sozluk_anahtarlari(self):
        veri = saf_ses_uret(500)
        f, g = fft_hesapla(veri, SAMPLE_RATE)
        sonuc = dominant_frekanslar(f, g)
        for item in sonuc:
            assert "frekans" in item
            assert "genlik" in item

    def test_440hz_dominant(self):
        veri = saf_ses_uret(440)
        f, g = fft_hesapla(veri, SAMPLE_RATE)
        sonuc = dominant_frekanslar(f, g, top_n=1)
        assert abs(sonuc[0]["frekans"] - 440.0) < 2.0


# ── Bant güç testleri ─────────────────────────────────────────────────────────

class TestBantGuc:

    def test_anahtarlar(self):
        veri = saf_ses_uret(1000)
        f, g = fft_hesapla(veri, SAMPLE_RATE)
        bantlar = bant_guc(f, g)
        assert set(bantlar.keys()) == {"bass", "mid", "treble", "yuksek"}

    def test_negatif_deger_yok(self):
        veri = saf_ses_uret(1000)
        f, g = fft_hesapla(veri, SAMPLE_RATE)
        bantlar = bant_guc(f, g)
        for v in bantlar.values():
            assert v >= 0

    def test_bass_sinyali(self):
        veri = saf_ses_uret(100)
        f, g = fft_hesapla(veri, SAMPLE_RATE)
        bantlar = bant_guc(f, g)
        assert bantlar["bass"] > bantlar["mid"]
        assert bantlar["bass"] > bantlar["treble"]

    def test_treble_sinyali(self):
        veri = saf_ses_uret(5000)
        f, g = fft_hesapla(veri, SAMPLE_RATE)
        bantlar = bant_guc(f, g)
        assert bantlar["treble"] > bantlar["bass"]


# ── dB dönüşüm testleri ───────────────────────────────────────────────────────

class TestDB:

    def test_cikti_tipi(self):
        veri = saf_ses_uret(440)
        f, g = fft_hesapla(veri, SAMPLE_RATE)
        db = db_donustur(g)
        assert isinstance(db, np.ndarray)

    def test_sifir_genligi(self):
        sifir = np.zeros(100)
        db = db_donustur(sifir)
        assert np.all(db < -100)


# ── Tam analiz testi ──────────────────────────────────────────────────────────

class TestAnalizEt:

    def test_sonuc_anahtarlari(self):
        veri = saf_ses_uret(440)
        sonuc = analiz_et(veri, SAMPLE_RATE)
        beklenen = {"frekanslar", "genlikler", "dominant", "bantlar", "db", "sample_rate", "sure"}
        assert beklenen.issub
