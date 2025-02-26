import os
import shutil
import random

# Diretório original onde estão as imagens organizadas por dígitos
root_dir = "../rede_neural_python_processamento_de_som/data/train_imgs/"

# Novo diretório onde as imagens de validação serão armazenadas
output_dir = "../rede_neural_python_processamento_de_som/data/valid_imgs/"
os.makedirs(output_dir, exist_ok=True)

# Percorre todas as pastas de dígitos (0 a 9)
for digit in range(10):
    digit_dir = os.path.join(root_dir, str(digit))
    
    # Lista todos os arquivos de imagem na pasta do dígito
    images = [img for img in os.listdir(digit_dir) if img.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # Calcula 20% do número total de imagens
    num_images_to_move = int(len(images) * 0.2)
    
    # Seleciona aleatoriamente 20% das imagens
    images_to_move = random.sample(images, num_images_to_move)
    
    # Cria a pasta de validação para o dígito se não existir
    validation_digit_dir = os.path.join(output_dir, str(digit))
    os.makedirs(validation_digit_dir, exist_ok=True)
    
    # Move as imagens selecionadas para a pasta de validação
    for img in images_to_move:
        src_path = os.path.join(digit_dir, img)
        dst_path = os.path.join(validation_digit_dir, img)
        shutil.move(src_path, dst_path)
        print(f"✅ Movido: {src_path} → {dst_path}")