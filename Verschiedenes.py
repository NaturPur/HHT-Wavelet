import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import pywt
import wave
import os

# Funktion zum Laden der WAV-Datei und Extrahieren der Metadaten
def load_wav(file_path, max_duration=None, amplitude_factor=1.0, highpass_cutoff=None):
    sample_rate, data = wavfile.read(file_path)
    with wave.open(file_path, 'rb') as wav_file:
        num_channels = wav_file.getnchannels()
        sample_rate_wave = wav_file.getframerate()
        bit_depth = wav_file.getsampwidth() * 8
        num_frames = wav_file.getnframes()
        duration = num_frames / sample_rate

    print(f"\nMetadaten der WAV-Datei: {file_path}")
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

    # Hochpassfilter anwenden, falls angegeben
    if highpass_cutoff is not None:
        nyquist = sample_rate / 2
        high = highpass_cutoff / nyquist
        b, a = signal.butter(4, high, btype='high')
        data = signal.filtfilt(b, a, data)
        print(f"Hochpassfilter bei {highpass_cutoff} Hz angewendet.")

    # Amplitudenverstärkung
    data = data * amplitude_factor
    print(f"Amplituden mit Faktor {amplitude_factor} verstärkt.")

    return sample_rate, data

# Funktion zur Durchführung der STFT
def compute_stft(data, sample_rate, nperseg=1024, noverlap=512):
    frequencies, times, Zxx = signal.stft(data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    return frequencies, times, Zxx

# Funktion zur Durchführung der Wavelet-Transformation
def compute_wavelet_transform(data, sample_rate, wavelet='cmor1.5-2.0', max_freq=20000):
    scales = np.arange(1, 128)
    frequencies = pywt.scale2frequency(wavelet, scales) * sample_rate
    print(f"Wavelet-Frequenzen (Hz): Min = {frequencies.min():.2f}, Max = {frequencies.max():.2f}")
    valid_scales = scales[frequencies <= max_freq]
    coefficients, frequencies = pywt.cwt(data, valid_scales, wavelet, sampling_period=1/sample_rate)
    return coefficients, frequencies

# Funktion zum Plotten der akustischen Energie über die Frequenz für drei Aufnahmen
def plot_energy_over_frequency(frequencies1, coefficients1, label1, frequencies2, coefficients2, label2, frequencies3, coefficients3, label3, freq_range=None, title_prefix=""):
    # Energie berechnen (Summe der Amplituden über die Zeit für jedes Frequenzband)
    coef_abs1 = np.abs(coefficients1)
    energy_over_freq1 = np.sum(coef_abs1, axis=1)
    coef_abs2 = np.abs(coefficients2)
    energy_over_freq2 = np.sum(coef_abs2, axis=1)
    coef_abs3 = np.abs(coefficients3)
    energy_over_freq3 = np.sum(coef_abs3, axis=1)

    # Frequenzbereich einschränken, falls angegeben
    if freq_range is not None:
        freq_min, freq_max = freq_range
        freq_mask1 = (frequencies1 >= freq_min) & (frequencies1 <= freq_max)
        freq_mask2 = (frequencies2 >= freq_min) & (frequencies2 <= freq_max)
        freq_mask3 = (frequencies3 >= freq_min) & (frequencies3 <= freq_max)
        frequencies1 = frequencies1[freq_mask1]
        energy_over_freq1 = energy_over_freq1[freq_mask1]
        frequencies2 = frequencies2[freq_mask2]
        energy_over_freq2 = energy_over_freq2[freq_mask2]
        frequencies3 = frequencies3[freq_mask3]
        energy_over_freq3 = energy_over_freq3[freq_mask3]
        title = f'{title_prefix} Akustische Energie über die Frequenz ({freq_min/1000:.0f}–{freq_max/1000:.0f} kHz)'
    else:
        title = f'{title_prefix} Akustische Energie über die Frequenz (0–20 kHz)'

    # Plot erstellen (Frequenz auf x-Achse, Energie auf y-Achse)
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies1, energy_over_freq1, label=label1, color='blue')
    plt.plot(frequencies2, energy_over_freq2, label=label2, color='orange')
    plt.plot(frequencies3, energy_over_freq3, label=label3, color='green')
    plt.title(title)
    plt.xlabel('Frequenz [Hz]')
    plt.ylabel('Energie')
    if freq_range is not None:
        plt.xlim(freq_range[0], freq_range[1])
    else:
        plt.xlim(0, 20000)
    plt.grid(True)
    plt.legend()
    plt.show()

    return (frequencies1, energy_over_freq1), (frequencies2, energy_over_freq2), (frequencies3, energy_over_freq3)

# Hauptfunktion
if __name__ == "__main__":
    # Pfade zu den WAV-Dateien
    file_path1 = r"C:\Users\denni\Desktop\Audiodaten\HKA\nonoise\mic_1m_0.0000g_s_20241218_101110.wav"
    file_path2 = r"C:\Users\denni\Desktop\Audiodaten\HKA\noise\outdoor parking lot\straight\free flow\co2\#2 Rundloch 1.00mm\mic_1m_0.2615g_s.wav"
    file_path3 = r"C:\Users\denni\Desktop\Audiodaten\HKA\noise\outdoor parking lot\straight\free flow\co2\#2 Rundloch 1.00mm\mic_1m_1.0016g_s.wav"  # Dritte Aufnahme

    # Labels aus den Dateinamen extrahieren
    label1 = os.path.basename(file_path1)
    label2 = os.path.basename(file_path2)
    label3 = os.path.basename(file_path3)

    # Laden der WAV-Dateien (Amplitudenverstärkung und Hochpassfilter)
    sample_rate1, data1 = load_wav(file_path1, max_duration=10, amplitude_factor=100.0, highpass_cutoff=1)
    sample_rate2, data2 = load_wav(file_path2, max_duration=10, amplitude_factor=100.0, highpass_cutoff=1)
    sample_rate3, data3 = load_wav(file_path3, max_duration=10, amplitude_factor=100.0, highpass_cutoff=1)

    # STFT berechnen
    frequencies_stft1, times_stft1, Zxx1 = compute_stft(data1, sample_rate1)
    frequencies_stft2, times_stft2, Zxx2 = compute_stft(data2, sample_rate2)
    frequencies_stft3, times_stft3, Zxx3 = compute_stft(data3, sample_rate3)

    # Wavelet-Transformation berechnen
    coefficients1, frequencies_wavelet1 = compute_wavelet_transform(data1, sample_rate1)
    coefficients2, frequencies_wavelet2 = compute_wavelet_transform(data2, sample_rate2)
    coefficients3, frequencies_wavelet3 = compute_wavelet_transform(data3, sample_rate3)

    # Akustische Energie über die Frequenz plotten (Wavelet, fokussiert auf 8–20 kHz)
    plot_energy_over_frequency(
        frequencies_wavelet1, coefficients1, label1,
        frequencies_wavelet2, coefficients2, label2,
        frequencies_wavelet3, coefficients3, label3,
        freq_range=(8000, 20000), title_prefix="Wavelet:"
    )

    # Akustische Energie über die Frequenz plotten (STFT, fokussiert auf 8–20 kHz)
    plot_energy_over_frequency(
        frequencies_stft1, Zxx1, label1,
        frequencies_stft2, Zxx2, label2,
        frequencies_stft3, Zxx3, label3,
        freq_range=(8000, 20000), title_prefix="STFT:"
    )

    # Optional: Energie über den gesamten Frequenzbereich plotten
    plot_energy_over_frequency(
        frequencies_wavelet1, coefficients1, label1,
        frequencies_wavelet2, coefficients2, label2,
        frequencies_wavelet3, coefficients3, label3,
        title_prefix="Wavelet (gesamter Bereich):"
    )
    plot_energy_over_frequency(
        frequencies_stft1, Zxx1, label1,
        frequencies_stft2, Zxx2, label2,
        frequencies_stft3, Zxx3, label3,
        title_prefix="STFT (gesamter Bereich):"
    )