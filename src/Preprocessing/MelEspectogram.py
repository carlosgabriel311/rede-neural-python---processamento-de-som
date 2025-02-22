import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def save_mel_spectrogram(audio_path, save_path):
    """Gera e salva um Mel Spectrogram a partir de um arquivo de áudio."""
    y, sr = librosa.load(audio_path, sr=16000)  # Carrega o áudio com 16 kHz
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(1, 1), dpi=32)  # 32x32 pixels
    librosa.display.specshow(mel_spec_db, sr=sr, cmap='inferno')
    plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Diretório onde estão as pastas dos atores (dados brutos)
root_dir = "../rede_neural_python_processamento_de_som/data/sort_audios/"

# Diretório de saída organizado por dígito
output_dir = "../rede_neural_python_processamento_de_som/data/imgs/"
os.makedirs(output_dir, exist_ok=True)

# Percorre todas as pastas de atores e arquivos .wav
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".wav"):  # Processa apenas arquivos de áudio
            audio_path = os.path.join(subdir, file)

            # Obtém o dígito do nome do arquivo (assumindo que o nome começa com o dígito)
            digit = file[0]  # Exemplo: "0_01_0.wav" → digit = "0"

            if digit.isdigit():  # Garante que seja um número entre 0 e 9
                output_folder = os.path.join(output_dir, digit)
                os.makedirs(output_folder, exist_ok=True)  # Cria a pasta do dígito

                # Nome do arquivo de imagem correspondente
                image_filename = os.path.splitext(file)[0] + ".png"
                save_path = os.path.join(output_folder, image_filename)

                # Gera e salva o espectrograma
                save_mel_spectrogram(audio_path, save_path)
                print(f"✅ Salvo: {save_path}")
