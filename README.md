# Real-Fake People Discriminator

Este projeto utiliza redes neurais profundas para distinguir rostos reais de rostos gerados por inteligência artificial (IA), com base em imagens. A solução foi construída com TensorFlow e combina diversos conjuntos de dados faciais para treinar um classificador binário robusto.

## 📁 Construção do Conjunto de Dados

O conjunto de dados foi construído a partir da união de três fontes principais, garantindo diversidade e melhor generalização:

- **[ThisPersonDoesNotExist.com](https://thispersondoesnotexist.com/)** – Rostos sintéticos gerados por IA.
- **[LFW (Labeled Faces in the Wild)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)** – Rostos reais de pessoas.
- **[140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)** – Conjunto com imagens reais e falsas separadas por pastas (train/val/test).

As imagens foram organizadas da seguinte forma:

data/
├── IA/ # Rostos gerados por IA <br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
└── Real/ # Rostos reais

## ⚙️ Pipeline de Treinamento

- Balanceamento das classes usando `tf.data.Dataset.sample_from_datasets`
- Pré-processamento com crop central, resize e normalização
- Aumento de dados com:
  - Flip horizontal
  - Alteração de brilho
  - Alteração de contraste
- Callbacks utilizados:
  - `EarlyStopping`
  - `ReduceLROnPlateau`
  - `ModelCheckpoint`

## 🚀 Suporte a GPU

O código detecta automaticamente se há GPU disponível:

```python
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Ativa a primeira GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        device = '/GPU:0'
        print("✅ Usando GPU")
    except RuntimeError as e:
        print("⚠️ Erro ao configurar GPU:", e)
        device = '/CPU:0'
else:
    print("❌ GPU não disponível, usando CPU")
    device = '/CPU:0'
```

## Exemplos de Imagens do Dataset e seus Rótulos

![Captura de tela de 2025-07-08 10-48-01](https://github.com/user-attachments/assets/9c9255cc-3229-4da8-9e43-800af7e2382b)

Em que:
- 1: Imagem gerada por IA;
- 0: Imagem Real.

## 🧪 Como Rodar

### 1. Clone o repositório

```bash
git clone https://github.com/PipInstallGustavo/Real-Fake-People-Discriminator.git
cd Real-Fake-People-Discriminator
```

### 2. Criar um Ambiente Virtual do Python
```bash
python3 -m venv ML_final
source ML_final/bin/activate  # No Windows: ML_final\Scripts\activate
```

### 3. Instale as dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Execute o notebook

**OBS:** Certifique-se de que você tenha as pastas data/IA e data/Real com as imagens organizadas corretamente antes de executar as células de treinamento. 


## Link para download do dataset
<a href="https://mega.nz/fm/qgx3WJpb"> https://mega.nz/fm/qgx3WJpb
