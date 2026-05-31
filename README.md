# Classificação de Câncer Pulmonar por Tomografia — Deep Learning com PyTorch

Projeto de aprendizado profundo para classificação automática de imagens de tomografia computadorizada do tórax em 4 categorias clínicas.

## Sobre o Dataset

| Split | Adenocarcinoma | Carcinoma G. Células | Normal | Carcinoma Escamoso | Total |
|-------|:--------------:|:--------------------:|:------:|:------------------:|:-----:|
| Treino | 195 | 115 | 148 | 155 | **613** |
| Validação | 23 | 21 | 13 | 15 | **72** |
| Teste | 120 | 51 | 54 | 90 | **315** |

## Estrutura do Projeto

```
chest/
├── Data/
│   ├── train/          # Conjunto de treinamento
│   ├── valid/          # Conjunto de validação
│   └── test/           # Conjunto de teste
│
├── notebooks/
│   ├── 01_exploracao_dados.ipynb     # EDA e análise visual
│   ├── 02_preprocessamento.ipynb    # Validação do pipeline
│   ├── 03_treinamento.ipynb         # Treinamento (2 fases)
│   └── 04_avaliacao.ipynb           # Métricas e análise de erros
│
├── src/
│   ├── config.py       # Configurações centrais (caminhos, hiperparâmetros)
│   ├── dataset.py      # Dataset PyTorch + DataLoaders
│   ├── modelo.py       # Arquitetura com Transfer Learning
│   ├── treinamento.py  # Loop de treino + Early Stopping
│   └── avaliacao.py    # Métricas, matriz de confusão, curvas ROC
│
├── modelos_salvos/     # Checkpoints (.pth)
├── logs/               # Logs do TensorBoard
├── reports/            # Gráficos e relatórios gerados
├── venv/               # Ambiente virtual Python 3.11
├── requirements.txt
└── README.md
```

## Configuração do Ambiente

### Pré-requisitos

- Python 3.11 (PyTorch não suporta Python 3.14+)
- macOS, Linux ou Windows

### Instalação

```bash
# 1. Clone ou acesse o diretório do projeto
cd chest

# 2. Crie e ative o ambiente virtual com Python 3.11
python3.11 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate.bat     # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Inicie o Jupyter
jupyter notebook notebooks/
```

## Fluxo dos Notebooks

Execute os notebooks na ordem:

| # | Notebook | O que faz |
|---|----------|-----------|
| 1 | `01_exploracao_dados.ipynb` | Análise exploratória: distribuição de classes, amostras visuais, dimensões |
| 2 | `02_preprocessamento.ipynb` | Verifica integridade, valida augmentações, calcula pesos de classe |
| 3 | `03_treinamento.ipynb` | Treina o modelo em 2 fases com Early Stopping |
| 4 | `04_avaliacao.ipynb` | Avalia no teste: acurácia, AUC-ROC, matriz de confusão, análise de erros |

## Arquitetura

- **Backbone:** ResNet50 pré-treinada no ImageNet (Transfer Learning)
- **Cabeça:** BatchNorm → Dropout → Linear(2048→256) → ReLU → Linear(256→4)
- **Estratégia de treino:**
  - Fase 1: backbone congelado, treina apenas a cabeça (10 épocas, LR=1e-3)
  - Fase 2: fine-tuning das últimas 30 camadas (até 30 épocas, LR=1e-4)
- **Loss:** CrossEntropyLoss com pesos por classe (combate desbalanceamento)
- **Otimizador:** Adam com weight decay L2
- **Scheduler:** ReduceLROnPlateau (reduz LR na estagnação)
- **Early Stopping:** paciência de 7 épocas

## Configurações Principais (`src/config.py`)

```python
ARQUITETURA_MODELO = "resnet50"   # Trocar por: resnet18, efficientnet_b0, densenet121
TAMANHO_LOTE       = 32
EPOCAS             = 30
TAXA_APRENDIZADO   = 1e-4
ALTURA_IMAGEM      = 224
LARGURA_IMAGEM     = 224
```

## Dependências Principais

| Pacote | Versão | Uso |
|--------|--------|-----|
| PyTorch | 2.2.x | Framework de deep learning |
| torchvision | 0.17.x | Modelos pré-treinados e transforms |
| scikit-learn | 1.5.x | Métricas de classificação |
| Pillow | — | Leitura e processamento de imagens |
| albumentations | 1.4.x | Augmentações avançadas (opcional) |
| matplotlib / seaborn | — | Visualizações |
| jupyter | — | Execução dos notebooks |
| tqdm | — | Barras de progresso |
| tensorboard | — | Monitoramento do treinamento |
