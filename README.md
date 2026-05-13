# FreqLab 🔊

Ses, titreşim ve nesnelerden gelen frekansları analiz eden açık kaynaklı Python projesi.

## Neler yapabilir?

- Mikrofon veya .wav/.mp3 dosyasından ses okur
- FFT ile frekans spektrumu çıkarır
- Dominant frekansları tespit eder
- Spektrogram ve dalga formu grafiği üretir
- Frekans bandı filtreleme yapar (bass, mid, treble)

## Kurulum

```bash
git clone https://github.com/KULLANICI_ADINIZ/freqlab.git
cd freqlab
pip install -r requirements.txt
```

## Kullanım

```bash
python src/analyzer.py --input data/samples/ornek.wav
```

## Proje Yapısı
