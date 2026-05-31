"""
config.py
---------
Arquivo central de configuração do projeto.
Define caminhos, hiperparâmetros e constantes utilizadas em todos os notebooks.
"""

import ssl
import certifi
from pathlib import Path

# Corrige erro de certificado SSL no macOS ao baixar pesos pré-treinados
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# ============================================================
# CAMINHOS DO PROJETO
# ============================================================

# Raiz do projeto (dois níveis acima deste arquivo: src/ -> chest/)
RAIZ_PROJETO = Path(__file__).resolve().parent.parent

# Caminhos dos dados
CAMINHO_DADOS       = RAIZ_PROJETO / "Data"
CAMINHO_TREINO      = CAMINHO_DADOS / "train"
CAMINHO_TESTE       = CAMINHO_DADOS / "test"
CAMINHO_VALIDACAO   = CAMINHO_DADOS / "valid"

# Caminhos de saída
CAMINHO_MODELOS     = RAIZ_PROJETO / "modelos_salvos"
CAMINHO_LOGS        = RAIZ_PROJETO / "logs"
CAMINHO_REPORTS     = RAIZ_PROJETO / "reports"

# Garante que os diretórios de saída existam
for _caminho in [CAMINHO_MODELOS, CAMINHO_LOGS, CAMINHO_REPORTS]:
    _caminho.mkdir(parents=True, exist_ok=True)

# ============================================================
# MAPEAMENTO DE CLASSES
# ============================================================

# Nomes simplificados das classes (usados no conjunto de teste)
CLASSES_TESTE = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma",
]

# Nomes das classes com descrição clínica completa (usados em treino/validação)
CLASSES_TREINO = [
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib",
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
    "normal",
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa",
]

# Mapeamento de nome completo → rótulo inteiro (índice da classe)
MAPA_CLASSES = {
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib":         0,
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa":      1,
    "normal":                                               2,
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa":   3,
    # Aliases para o conjunto de teste (nomes curtos)
    "adenocarcinoma":          0,
    "large.cell.carcinoma":    1,
    "squamous.cell.carcinoma": 3,
}

# Lista de rótulos legíveis para exibição em gráficos e relatórios
NOMES_CLASSES = [
    "Adenocarcinoma",
    "Carcinoma de Células Grandes",
    "Normal",
    "Carcinoma de Células Escamosas",
]

NUM_CLASSES = len(NOMES_CLASSES)

# ============================================================
# HIPERPARÂMETROS DE PRÉ-PROCESSAMENTO
# ============================================================

# Dimensões de entrada da rede (altura x largura em pixels)
ALTURA_IMAGEM  = 224
LARGURA_IMAGEM = 224

# Estatísticas de normalização ImageNet (utilizadas em transfer learning)
MEDIA_IMAGENET  = [0.485, 0.456, 0.406]
DESVIO_IMAGENET = [0.229, 0.224, 0.225]

# ============================================================
# HIPERPARÂMETROS DE TREINAMENTO
# ============================================================

TAMANHO_LOTE     = 32        # Número de imagens por batch
EPOCAS           = 30        # Número máximo de épocas de treinamento
TAXA_APRENDIZADO = 1e-4      # Learning rate inicial
PESO_DECAIMENTO  = 1e-4      # Regularização L2 (weight decay)
PACIENCIA        = 7         # Épocas sem melhora antes do Early Stopping
SEMENTE_ALEATORIA = 42       # Semente para reprodutibilidade

# ============================================================
# CONFIGURAÇÕES DO MODELO
# ============================================================

# Arquitetura backbone para transfer learning
# Opções: "resnet50", "resnet18", "efficientnet_b0", "densenet121"
ARQUITETURA_MODELO = "resnet50"

# Se True, congela os pesos do backbone durante o fine-tuning inicial
CONGELAR_BACKBONE = True

# Nome do arquivo onde o melhor modelo será salvo
NOME_ARQUIVO_MODELO = f"melhor_modelo_{ARQUITETURA_MODELO}.pth"
