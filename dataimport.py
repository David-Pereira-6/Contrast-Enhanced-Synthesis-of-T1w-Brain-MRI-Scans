import matplotlib.pyplot as plt
import pandas as pd
import os
import zipfile
import shutil
import nibabel as nib
import ipywidgets as widgets
import cv2
import plotly.graph_objects as go
from skimage import measure



data_path = 'C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/LGG.zip'
Unziped_data_path = 'C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/LGG_data'
usable_data_path = 'C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/usable_data'

os.makedirs(Unziped_data_path, exist_ok=True)

with zipfile.ZipFile(data_path, 'r') as zip_ref:
    zip_ref.extractall(Unziped_data_path)

excel_file = pd.read_excel(os.path.join(Unziped_data_path, 'TumorType_Grade.xlsx'))

images = ['t1', 't1Gd', 't2']

# Iterar sobre cada paciente (pasta)
for patient in os.listdir(Unziped_data_path):
    patient_path = os.path.join(Unziped_data_path, patient)
    if os.path.isdir(patient_path):
        dest_dir = os.path.join(usable_data_path, patient)
        os.makedirs(dest_dir, exist_ok=True)

        # Verificar os ficheiros dentro de cada pasta
        for file in os.listdir(patient_path):
            for modality in images:
                if modality.lower() in file.lower():
                    nii_path = os.path.join(patient_path, file)

                    # Carregar volume completo
                    img = nib.load(nii_path)
                    data = img.get_fdata()
                    img_size = data.shape[0]
                    print(img_size)
                    affine = img.affine

                    # Guardar novo ficheiro .nii (com o volume todo)
                    output_filename = os.path.splitext(os.path.splitext(file)[0])[0] + '.nii'
                    output_path = os.path.join(dest_dir, output_filename)
                    nib.save(nib.Nifti1Image(data, affine), output_path)


# shutil.rmtree('C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/usable_data') #apagar o usable data NAOO CORRER, so quando dá merda


# Carregar o volume
nii_path = 'C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/usable_data/TCGA-CS-4942/TCGA-CS-4942_1997.02.22_t1.nii'  # Substitua pelo caminho correto
img = nib.load(nii_path)
data = img.get_fdata()

# Configurar parâmetros do vídeo
video_path = 'slices_video.mp4'  # Nome do arquivo de vídeo
fps = 10  # Frames por segundo
height, width = data.shape[0], data.shape[1]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec de vídeo

# Criar escritor de vídeo
out = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=False)

# Percorrer as slices e criar frames
for i in range(data.shape[2]):
    # Gerar a imagem da slice
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(data[:, :, i], cmap='gray')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('temp_slice.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Carregar o frame como uma imagem
    frame = cv2.imread('temp_slice.png', cv2.IMREAD_GRAYSCALE)
    frame = cv2.resize(frame, (width, height))  # Ajustar o tamanho
    out.write(frame)

# Liberar recursos
out.release()
print(f"Vídeo salvo em: {video_path}")


# Caminho do diretório principal (substitua pelo caminho correto)
diretorio_principal = "C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/usable_data"
diretorio_t1 = "C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/t1"
diretorio_t1Gd = "C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/t1Gd"
diretorio_t2 = "C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/t2"

# Cria os diretórios para cada categoria, caso não existam
os.makedirs(diretorio_t1, exist_ok=True)
os.makedirs(diretorio_t1Gd, exist_ok=True)
os.makedirs(diretorio_t2, exist_ok=True)

for pasta_raiz, subpastas, arquivos in os.walk(diretorio_principal):
    for arquivo in arquivos:
        caminho_arquivo = os.path.join(pasta_raiz, arquivo)

        if "t1gd" in arquivo.lower():  # Verifica se é um arquivo t1Gd
            destino = os.path.join(diretorio_t1Gd, arquivo)
            shutil.copy2(caminho_arquivo, destino)
        elif "t1" in arquivo.lower():  # Verifica se é um arquivo t1
            destino = os.path.join(diretorio_t1, arquivo)
            shutil.copy2(caminho_arquivo, destino)
        elif "t2" in arquivo.lower():  # Verifica se é um arquivo t2
            destino = os.path.join(diretorio_t2, arquivo)
            shutil.copy2(caminho_arquivo, destino)

print(f"Arquivos organizados nas pastas:\n- {diretorio_t1}\n- {diretorio_t1Gd}\n- {diretorio_t2}")

