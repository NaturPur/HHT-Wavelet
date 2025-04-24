# Version: 1.7.0

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
    'stft_bandpass_low': 1,  # Untere Grenze des Bandpassfilters für STFT (Hz)
    'stft_bandpass_high': 2000,  # Obere Grenze des Bandpassfilters für STFT (Hz)
    'wavelet_bandpass_low': 1,  # Untere Grenze des Bandpassfilters für Wavelet (Hz)
    'wavelet_bandpass_high': 2000,  # Obere Grenze des Bandpassfilters für Wavelet (Hz)
    'amplitude_factor': 100.0,  # Amplitudenverstärkungsfaktor
    'wavelet_type': 'cmor1.5-2.0',  # Wavelet-Typ für die Wavelet-Transformation (angepasst, um Warnung zu vermeiden)
    'wavelet_scales': np.arange(1, 64),  # Skalen für die Wavelet-Transformation (reduziert für bessere Performance)
    'stft_nperseg': 2048,  # Fenstergröße für die STFT (erhöht für bessere Frequenzauflösung)
    'stft_noverlap': 512,  # Überlappung für die STFT (angepasst an nperseg)
    'max_duration': 10,  # Maximale Dauer der Aufnahme in Sekunden (None für gesamte Datei)
    'freq_range': (1, 2000),  # Frequenzbereich für die Visualisierung (Hz)
    'num_processes': 5,  # Anzahl der Prozesse für Multiprocessing (reduziert für bessere Stabilität)
}


# Funktion zum Laden der WAV-Datei und Extrahieren der Metadaten
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
def compute_wavelet_transform(data, sample_rate, wavelet='cmor1.5-2.0', scales=np.arange(1, 64), max_freq=20000,
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


# Funktion zum Plotten der akustischen Energie über die Frequenz für Wavelet und STFT in einem Diagramm
def plot_energy_over_frequency(frequencies_wavelet, coefficients_wavelet, frequencies_stft, coefficients_stft,
                               freq_range=None):
    # Energie für Wavelet berechnen
    coef_abs_wavelet = np.abs(coefficients_wavelet)
    energy_wavelet = np.sum(coef_abs_wavelet ** 2, axis=1)  # Summe über die Zeit
    if energy_wavelet.max() > 0:
        energy_wavelet = energy_wavelet / energy_wavelet.max()
    else:
        print("Warnung: Maximale Energie für Wavelet ist 0")

    # Energie für STFT berechnen
    coef_abs_stft = np.abs(coefficients_stft)
    energy_stft = np.sum(coef_abs_stft ** 2, axis=1)  # Summe über die Zeit
    if energy_stft.max() > 0:
        energy_stft = energy_stft / energy_stft.max()
    else:
        print("Warnung: Maximale Energie für STFT ist 0")

    # Debug: Energie überprüfen
    print(f"Wavelet Energie Min: {energy_wavelet.min():.2e}, Max: {energy_wavelet.max():.2e}")
    print(f"STFT Energie Min: {energy_stft.min():.2e}, Max: {energy_stft.max():.2e}")

    # Frequenzbereich für Wavelet einschränken, falls angegeben
    if freq_range is not None:
        freq_min, freq_max = freq_range
        freq_mask_wavelet = (frequencies_wavelet >= freq_min) & (frequencies_wavelet <= freq_max)
        frequencies_wavelet = frequencies_wavelet[freq_mask_wavelet]
        energy_wavelet = energy_wavelet[freq_mask_wavelet]
    else:
        freq_min, freq_max = 0, 20000

    # Frequenzbereich für STFT einschränken, falls angegeben
    if freq_range is not None:
        freq_mask_stft = (frequencies_stft >= freq_min) & (frequencies_stft <= freq_max)
        frequencies_stft = frequencies_stft[freq_mask_stft]
        energy_stft = energy_stft[freq_mask_stft]

    # Plot erstellen
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Wavelet auf der linken y-Achse plotten (Blau)
    ax1.plot(frequencies_wavelet, energy_wavelet, color='blue', label='Wavelet')
    ax1.set_xlabel('Frequenz [Hz]')
    ax1.set_ylabel('Energie (Wavelet, normalisiert)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    ax1.set_xlim(freq_min, freq_max)

    # STFT auf der rechten y-Achse plotten (Rot)
    ax2 = ax1.twinx()
    ax2.plot(frequencies_stft, energy_stft, color='red', label='STFT')
    ax2.set_ylabel('Energie (STFT, normalisiert)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Dynamische Skalierung der y-Achsen
    max_energy_wavelet = energy_wavelet.max()
    max_energy_stft = energy_stft.max()
    if max_energy_wavelet > 0:
        ax1.set_ylim(0, max_energy_wavelet * 1.1)  # 10% Spielraum
    else:
        ax1.set_ylim(0, 1)
    if max_energy_stft > 0:
        ax2.set_ylim(0, max_energy_stft * 1.1)  # 10% Spielraum
    else:
        ax2.set_ylim(0, 1)

    # Titel und Legende
    plt.title(f'Akustische Energie über die Frequenz ({freq_min / 1000:.0f}–{freq_max / 1000:.0f} kHz)')
    # Kombinierte Legende
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    return fig


# Funktion zur Verarbeitung einer einzelnen WAV-Datei (für Multiprocessing)
def process_single_wav(file_info, input_dir, output_dir):
    try:
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

            # Energie über die Frequenz plotten (Wavelet und STFT kombiniert)
            fig = plot_energy_over_frequency(
                frequencies_wavelet, coefficients,
                frequencies_stft, Zxx,
                freq_range=CONFIG['freq_range']
            )
            output_path = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}_combined_energy.png")
            fig.savefig(output_path)
            plt.close(fig)
            print(f"Kombiniertes Diagramm (Wavelet & STFT) gespeichert: {output_path}")
    except KeyboardInterrupt:
        print(f"Prozess für Datei {file} abgebrochen (KeyboardInterrupt).")
        raise
    except Exception as e:
        print(f"Fehler bei der Verarbeitung von Datei {file}: {str(e)}")
        raise


# Hauptfunktion mit Multiprocessing
def process_wav_files(input_dir, output_dir):
    # Liste aller WAV-Dateien sammeln
    wav_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append((root, file))

    # Anzahl der Prozesse (reduziert für bessere Stabilität)
    num_processes = CONFIG['num_processes']
    print(f"Verwende {num_processes} Prozesse für die Verarbeitung.")

    # Pool für Multiprocessing erstellen
    try:
        with mp.Pool(processes=num_processes) as pool:
            # Teilverarbeitungsfunktion mit input_dir und output_dir fixieren
            process_func = partial(process_single_wav, input_dir=input_dir, output_dir=output_dir)
            # Parallele Verarbeitung der WAV-Dateien
            pool.map(process_func, wav_files)
    except KeyboardInterrupt:
        print("Multiprocessing abgebrochen (KeyboardInterrupt).")
        pool.terminate()
        pool.join()
        raise
    except Exception as e:
        print(f"Fehler im Multiprocessing: {str(e)}")
        pool.terminate()
        pool.join()
        raise


# Hauptfunktion
if __name__ == "__main__":
    # Eingabe- und Ausgabeverzeichnis
    input_dir = r"C:\Users\denni\Desktop\Audiodaten\HKA\nonoise"
    output_dir = r"C:\Users\denni\Desktop\Audiodaten\Wavelet"

    # Verzeichnisstruktur durchlaufen und Diagramme erstellen
    process_wav_files(input_dir, output_dir)