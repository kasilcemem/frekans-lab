# FreqLab

Ses, titreşim ve nesnelerden gelen frekansları analiz eden açık kaynaklı Python projesi.
Mikrofon girişi, .wav ve .mp3 dosyalarını okur; FFT ile spektrum çıkarır, dominant frekansları
tespit eder, filtreler uygular ve grafiklerle görselleştirir.

---

## Özellikler

- Mikrofon veya dosyadan ses okuma (.wav, .mp3)
- FFT ile frekans spektrumu hesaplama
- Dominant frekans tespiti
- Bass / mid / treble / yüksek bant güç analizi
- Alçak geçiren, yüksek geçiren, bant geçiren filtreler
- Frekansı müzik notasına çevirme (440 Hz → A4)
- Dalga formu, spektrum, dB grafiği, mel spektrogramı
- Bant güç dağılımı bar grafiği
- Tüm modüller için pytest birim testleri

---

## Kurulum

```bash
