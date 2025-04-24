# Version: 1.5.0

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import pywt
import wave
import os
import multiprocessing as mp
from functools import partial

# Einstellbare Parameter (oben im Skript)
CONFIG = {
    'stft_bandpass_low': 500,  # Untere Grenze des Bandpassfilters für STFT (Hz)
    'stft_bandpass_high': 20000,  # Obere Grenze des Bandpassfilters für STFT (Hz)
    'wavelet_bandpass_low': 1,  # Untere Grenze des Bandpassfilters für Wavelet (Hz)
    'wavelet_bandpass_high': 20000,  # Obere Grenze des Bandpassfilters für Wavelet (Hz)
    'amplitude_factor': 100.0,  # Amplitudenverstärkungsfaktor
    'wavelet_type': 'cmor1.5-2.0',  # Wavelet-Typ für die Wavelet-Transformation
    'wavelet_scales': np.arange(1, 128),  # Skalen für die Wavelet-Transformation
    'stft_nperseg': 2048,  # Fenstergröße für die STFT (erhöht für bessere Frequenzauflösung)
    'stft_noverlap': 1024,  # Überlappung für die STFT (angepasst an nperseg)
    'max_duration': 10,  # Maximale Dauer der Aufnahme in Sekunden (None für gesamte Datei)
    'freq_range': (1, 20000),  # Frequenzbereich für die Visualisierung (Hz)
}


# Funktion zum Laden der WAV-datei und Extrahieren der Metadaten
def load_wav(file_path, max_duration=None, amplitude_factor=1.0):
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

    # Amplitudenverstärkung
    data = data * amplitude_factor
    print(f"Amplituden mit Faktor {amplitude_factor} verstärkt.")

    return sample_rate, data


# Funktion zur Durchführung der STFT
def compute_stft(data, sample_rate, nperseg=1024, noverlap=512, bandpass_low=None, bandpass_high=None):
    # Bandpassfilter anwenden, falls angegeben
    if bandpass_low is not None and bandpass_high is not None:
        nyquist = sample_rate / 2
        low = bandpass_low / nyquist
        high = bandpass_high / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        data = signal.filtfilt(b, a, data)
        print(f"STFT: Bandpassfilter von {bandpass_low} Hz bis {bandpass_high} Hz angewendet.")

    frequencies, times, Zxx = signal.stft(data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    # Debug: Amplituden überprüfen
    print(f"STFT Amplituden (Zxx) Min: {np.abs(Zxx).min():.2e}, Max: {np.abs(Zxx).max():.2e}")
    return frequencies, times, Zxx


# Funktion zur Durchführung der Wavelet-Transformation
def compute_wavelet_transform(data, sample_rate, wavelet='cmor1.5-2.0', scales=np.arange(1, 128), max_freq=20000,
                              bandpass_low=None, bandpass_high=None):
    # Bandpassfilter anwenden, falls angegeben
    if bandpass_low is not None and bandpass_high is not None:
        nyquist = sample_rate / 2
        low = bandpass_low / nyquist
        high = bandpass_high / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        data = signal.filtfilt(b, a, data)
        print(f"Wavelet: Bandpassfilter von {bandpass_low} Hz bis {bandpass_high} Hz angewendet.")

    frequencies = pywt.scale2frequency(wavelet, scales) * sample_rate
    print(f"Wavelet-Frequenzen (Hz): Min = {frequencies.min():.2f}, Max = {frequencies.max():.2f}")
    valid_scales = scales[frequencies <= max_freq]
    coefficients, frequencies = pywt.cwt(data, valid_scales, wavelet, sampling_period=1 / sample_rate)
    # Debug: Amplituden überprüfen
    print(
        f"Wavelet Amplituden (coefficients) Min: {np.abs(coefficients).min():.2e}, Max: {np.abs(coefficients).max():.2e}")
    return coefficients, frequencies


# Funktion zum Plotten der akustischen Energie über die Frequenz für eine Datei
def plot_energy_over_frequency(frequencies, coefficients, freq_range=None, title_prefix=""):
    # Energie berechnen (Summe der quadrierten Amplituden über die Zeit für jedes Frequenzband)
    coef_abs = np.abs(coefficients)
    # Energie als Leistung (Amplitude^2) berechnen
    energy_over_freq = np.sum(coef_abs ** 2, axis=1)  # Summe über die Zeit
    # Normalisierung der Energie (auf Maximum = 1)
    if energy_over_freq.max() > 0:
        energy_over_freq = energy_over_freq / energy_over_freq.max()
    else:
        print(f"Warnung: Maximale Energie ist 0 bei {title_prefix}")

    # Debug: Energie überprüfen
    print(f"{title_prefix} Energie Min: {energy_over_freq.min():.2e}, Max: {energy_over_freq.max():.2e}")

    # Frequenzbereich einschränken, falls angegeben
    if freq_range is not None:
        freq_min, freq_max = freq_range
        freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
        frequencies = frequencies[freq_mask]
        energy_over_freq = energy_over_freq[freq_mask]
        title = f'{title_prefix} Akustische Energie über die Frequenz ({freq_min / 1000:.0f}–{freq_max / 1000:.0f} kHz)'
    else:
        title = f'{title_prefix} Akustische Energie über die Frequenz (0–20 kHz)'

    # Plot erstellen (Frequenz auf x-Achse, Energie auf y-Achse)
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, energy_over_freq)
    plt.title(title)
    plt.xlabel('Frequenz [Hz]')
    plt.ylabel('Energie (normalisiert)')
    if freq_range is not None:
        plt.xlim(freq_range[0], freq_range[1])
    else:
        plt.xlim(0, 20000)
    plt.grid(True)
    # y-Achse dynamisch skalieren (automatisch basierend auf den Daten)
    max_energy = energy_over_freq.max()
    if max_energy > 0:
        plt.ylim(0, max_energy * 1.1)  # 10% Spielraum über dem Maximum
    else:
        plt.ylim(0, 1)  # Fallback, falls max_energy 0 ist
    return plt.gcf()


# Funktion zur Verarbeitung einer einzelnen WAV-Datei (für Multiprocessing)
def process_single_wav(file_info, input_dir, output_dir):
    root, file = file_info
    if file.lower().endswith('.wav'):
        # Pfad zur WAV-Datei
        wav_path = os.path.join(root, file)

        # Zielpfad für das Diagramm (Struktur ab 'Audiodaten' nachbilden)
        audiodaten_index = root.find('Audiodaten')
        if audiodaten_index == -1:
            raise ValueError(f"Verzeichnis 'Audiodaten' nicht im Pfad {root} gefunden.")

        # Extrahiere den relativen Pfad ab 'Audiodaten'
        relative_path = root[audiodaten_index:]
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        # WAV-Datei laden und transformieren
        sample_rate, data = load_wav(
            wav_path,
            max_duration=CONFIG['max_duration'],
            amplitude_factor=CONFIG['amplitude_factor']
        )

        # STFT berechnen
        frequencies_stft, times_stft, Zxx = compute_stft(
            data, sample_rate,
            nperseg=CONFIG['stft_nperseg'],
            noverlap=CONFIG['stft_noverlap'],
            bandpass_low=CONFIG['stft_bandpass_low'],
            bandpass_high=CONFIG['stft_bandpass_high']
        )

        # Wavelet-Transformation berechnen
        coefficients, frequencies_wavelet = compute_wavelet_transform(
            data, sample_rate,
            wavelet=CONFIG['wavelet_type'],
            scales=CONFIG['wavelet_scales'],
            bandpass_low=CONFIG['wavelet_bandpass_low'],
            bandpass_high=CONFIG['wavelet_bandpass_high']
        )

        # Energie über die Frequenz plotten (Wavelet)
        fig = plot_energy_over_frequency(
            frequencies_wavelet, coefficients,
            freq_range=CONFIG['freq_range'],
            title_prefix="Wavelet:"
        )
        output_path = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}_wavelet_energy.png")
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Wavelet-Diagramm gespeichert: {output_path}")

        # Energie über die Frequenz plotten (STFT)
        fig = plot_energy_over_frequency(
            frequencies_stft, Zxx,
            freq_range=CONFIG['freq_range'],
            title_prefix="STFT:"
        )
        output_path = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}_stft_energy.png")
        fig.savefig(output_path)
        plt.close(fig)
        print(f"STFT-Diagramm gespeichert: {output_path}")


# Hauptfunktion mit Multiprocessing
def process_wav_files(input_dir, output_dir):
    # Liste aller WAV-Dateien sammeln
    wav_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append((root, file))

    # Anzahl der Prozesse (Standard: Anzahl der CPU-Kerne)
    num_processes = mp.cpu_count()
    print(f"Verwende {num_processes} Prozesse für die Verarbeitung.")

    # Pool für Multiprocessing erstellen
    with mp.Pool(processes=num_processes) as pool:
        # Teilverarbeitungsfunktion mit input_dir und output_dir fixieren
        process_func = partial(process_single_wav, input_dir=input_dir, output_dir=output_dir)
        # Parallele Verarbeitung der WAV-Dateien
        pool.map(process_func, wav_files)


# Hauptfunktion
if __name__ == "__main__":
    # Eingabe- und Ausgabeverzeichnis
    input_dir = r"C:\Users\denni\Desktop\Audiodaten\HKA\noise\outdoor parking lot\straight\free flow\co2\#2 Rundloch 1.00mm"
    output_dir = r"C:\Users\denni\Desktop\Audiodaten\Wavelet"

    # Verzeichnisstruktur durchlaufen und Diagramme erstellen
    process_wav_files(input_dir, output_dir)