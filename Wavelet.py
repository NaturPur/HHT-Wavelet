import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import pywt
import wave

# Funktion zum Laden der WAV-Datei und Extrahieren der Metadaten
def load_wav(file_path, max_duration=None):
    sample_rate, data = wavfile.read(file_path)
    with wave.open(file_path, 'rb') as wav_file:
        num_channels = wav_file.getnchannels()
        sample_rate_wave = wav_file.getframerate()
        bit_depth = wav_file.getsampwidth() * 8
        num_frames = wav_file.getnframes()
        duration = num_frames / sample_rate

    print(f"Metadaten der WAV-Datei: {file_path}")
    print(f"Abtastrate: {sample_rate} Hz")
    print(f"Anzahl der Kanäle: {num_channels}")
    print(f"Bit-Tiefe: {bit_depth} Bit")
    print(f"Dauer: {duration:.2f} Sekunden")
    print(f"Total Samples: {num_frames}")

    if len(data.shape) > 1:
        data = data[:, 0]

    if max_duration is not None:
        max_samples = int(max_duration * sample_rate)
        data = data[:max_samples]
        duration = min(duration, max_duration)
        num_frames = len(data)
        print(f"\nSignal auf {max_duration} Sekunden gekürzt:")
        print(f"Neue Dauer: {duration:.2f} Sekunden")
        print(f"Neue Anzahl Samples: {num_frames}")

    return sample_rate, data

# Funktion zur Durchführung der STFT
def compute_stft(data, sample_rate, nperseg=512):
    frequencies, times, Zxx = signal.stft(data, fs=sample_rate, nperseg=nperseg)
    return frequencies, times, Zxx

# Funktion zur Durchführung der Wavelet-Transformation
def compute_wavelet_transform(data, sample_rate, wavelet='cmor1.5-1.0', max_freq=20000):
    scales = np.arange(1, 128)
    frequencies = pywt.scale2frequency(wavelet, scales) * sample_rate
    print(f"Wavelet-Frequenzen (Hz): Min = {frequencies.min():.2f}, Max = {frequencies.max():.2f}")
    valid_scales = scales[frequencies <= max_freq]
    coefficients, frequencies = pywt.cwt(data, valid_scales, wavelet, sampling_period=1/sample_rate)
    return coefficients, frequencies

# Funktion zum Plotten der Spektrogramme ohne logarithmische Skalierung
def plot_spectrograms(times_stft, frequencies_stft, Zxx, times_wavelet, frequencies_wavelet, coefficients, sample_rate):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # STFT Spektrogramm mit rohen Amplituden und kontrastreicher Colormap
    Zxx_abs = np.abs(Zxx)
    vmin_stft = np.percentile(Zxx_abs, 10)  # Untere Grenze (10. Perzentil)
    vmax_stft = np.percentile(Zxx_abs, 99)  # Obere Grenze (99. Perzentil)
    im1 = ax1.pcolormesh(times_stft, frequencies_stft, Zxx_abs, shading='gouraud', cmap='hot', vmin=vmin_stft, vmax=vmax_stft)
    ax1.set_title('STFT Spektrogramm')
    ax1.set_ylabel('Frequenz [Hz]')
    ax1.set_xlabel('Zeit [s]')
    ax1.set_ylim(0, 20000)
    plt.colorbar(im1, ax=ax1, label='Amplitude')

    # Wavelet Spektrogramm mit rohen Amplituden und kontrastreicher Colormap
    coef_abs = np.abs(coefficients)
    vmin_wavelet = np.percentile(coef_abs, 10)  # Untere Grenze (10. Perzentil)
    vmax_wavelet = np.percentile(coef_abs, 99)  # Obere Grenze (99. Perzentil)
    times_wavelet = np.linspace(0, len(data)/sample_rate, coefficients.shape[1])
    im2 = ax2.pcolormesh(times_wavelet, frequencies_wavelet, coef_abs, shading='gouraud', cmap='hot', vmin=vmin_wavelet, vmax=vmax_wavelet)
    ax2.set_title('Wavelet Spektrogramm')
    ax2.set_ylabel('Frequenz [Hz]')
    ax2.set_xlabel('Zeit [s]')
    ax2.set_ylim(0, 20000)
    plt.colorbar(im2, ax=ax2, label='Amplitude')

    plt.tight_layout()
    plt.show()

# Hauptfunktion
if __name__ == "__main__":
    # Pfad zur WAV-Datei
    file_path = r"C:\Users\Dennis\Desktop\mic_1m_0.8841g_s.wav"

    # Laden der WAV-Datei
    sample_rate, data = load_wav(file_path, max_duration=10)

    # STFT berechnen
    frequencies_stft, times_stft, Zxx = compute_stft(data, sample_rate)

    # Wavelet-Transformation berechnen
    coefficients, frequencies_wavelet = compute_wavelet_transform(data, sample_rate)

    # Spektrogramme plotten
    plot_spectrograms(times_stft, frequencies_stft, Zxx, None, frequencies_wavelet, coefficients, sample_rate)