import zipfile
import os

# Caminho para o arquivo ZIP
zip_path = "Master-FoundationModels-main.zip"

# Caminho para o diretório onde o ZIP está (mesmo diretório do script)
output_dir = os.path.dirname(zip_path)

# Abrir e descompactar o arquivo ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

print(f"Arquivos descompactados em: {output_dir}")
