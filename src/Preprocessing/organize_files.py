import os
import shutil

# Diretório original onde estão os áudios organizados por atores
root_dir = "../rede_neural_python_processamento_de_som/data/audios/"

# Novo diretório onde os arquivos serão organizados por dígito
output_dir = "../rede_neural_python_processamento_de_som/data/sort_audios/"
os.makedirs(output_dir, exist_ok=True)

# Percorre todas as pastas de atores e arquivos de áudio
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".wav"):  # Processa apenas arquivos .wav
            audio_path = os.path.join(subdir, file)

            # Obtém o dígito falado (assumindo que o nome do arquivo começa com o dígito)
            digit = file[0]  # Exemplo: "0_01_0.wav" → digit = "0"

            if digit.isdigit():  # Garante que seja um número entre 0 e 9
                output_folder = os.path.join(output_dir, digit)
                os.makedirs(output_folder, exist_ok=True)  # Cria a pasta do dígito se não existir

                # Define o novo caminho para o arquivo
                new_audio_path = os.path.join(output_folder, file)

                # Move o arquivo para a nova pasta
                shutil.move(audio_path, new_audio_path)
                print(f"✅ Movido: {audio_path} → {new_audio_path}")
