import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pytest

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
    t = np.linspace(0, sure, int(sample_rate * sure), endpoint=False)
    return np.sin(2 * np.pi * frekans * t).astype(np.float32)


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
        assert abs(en_guclu - hedef) < 2.0

    def test_sessizlik(self):
        veri = np.zeros(SAMPLE_RATE, dtype=np.float32)
        frekanslar, genlikler = fft_hesapla(veri, SAMPLE_RATE)
        assert genlikler.max() < 1e-6


class
