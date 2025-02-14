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

    plt.figure(figsize=(2, 2), dpi=64)  # Gera imagem 128x128
    librosa.display.specshow(mel_spec_db, sr=sr, cmap='inferno')
    plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Diretório onde estão as pastas de áudio
root_dir = "../rede_neural_python_processamento_de_som/data/audios/"

# Diretório para salvar as imagens
output_dir = "../rede_neural_python_processamento_de_som/data/imgs/"
os.makedirs(output_dir, exist_ok=True)  # Cria a pasta de destino se não existir

# Percorre todas as pastas e arquivos dentro de `root_dir`
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".wav"):  # Processa apenas arquivos .wav
            audio_path = os.path.join(subdir, file)

            # Cria um caminho de saída correspondente para a imagem
            relative_path = os.path.relpath(subdir, root_dir)  # Nome relativo da subpasta
            output_folder = os.path.join(output_dir, relative_path)
            os.makedirs(output_folder, exist_ok=True)  # Garante que a pasta exista

            # Nome do arquivo de imagem correspondente
            image_filename = os.path.splitext(file)[0] + ".png"
            save_path = os.path.join(output_folder, image_filename)

            # Gera e salva o espectrograma
            save_mel_spectrogram(audio_path, save_path)
            print(f"✅ Salvo: {save_path}")

