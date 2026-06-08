"""
config.py
---------
Configuração central do projeto de classificação de câncer pulmonar.

Este módulo é a única fonte de verdade para caminhos, hiperparâmetros e
constantes. Todos os demais módulos importam daqui — nunca defina valores
de configuração localmente em notebooks ou outros arquivos.

Exportações principais
----------------------
Caminhos       : CAMINHO_TREINO, CAMINHO_VALIDACAO, CAMINHO_TESTE,
                 CAMINHO_MODELOS, CAMINHO_LOGS,
                 CAMINHO_REPORTS_EDA, CAMINHO_REPORTS_TREINO, CAMINHO_REPORTS_AVALIACAO
Classes        : NOMES_CLASSES, MAPA_CLASSES, NUM_CLASSES
Hiperparâmetros: TAMANHO_LOTE, EPOCAS, TAXA_APRENDIZADO, PESO_DECAIMENTO,
                 PACIENCIA, SEMENTE_ALEATORIA
Modelo         : ARQUITETURA_MODELO, CONGELAR_BACKBONE, NOME_ARQUIVO_MODELO

Uso
---
    from config import CAMINHO_TREINO, NUM_CLASSES, TAMANHO_LOTE
"""

import ssl
import certifi
from pathlib import Path

# Corrige a verificação de certificado SSL no macOS, necessária para que o
# torchvision consiga baixar os pesos pré-treinados do ImageNet.
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

# ============================================================
# CAMINHOS DO PROJETO
# ============================================================

# Raiz do projeto (src/ -> chest/)
RAIZ_PROJETO = Path(__file__).resolve().parent.parent

# Conjuntos de dados
CAMINHO_DADOS     = RAIZ_PROJETO / "Data"
CAMINHO_TREINO    = CAMINHO_DADOS / "train"
CAMINHO_VALIDACAO = CAMINHO_DADOS / "valid"
CAMINHO_TESTE     = CAMINHO_DADOS / "test"

# Artefatos do treinamento
CAMINHO_MODELOS = RAIZ_PROJETO / "modelos_salvos"
CAMINHO_LOGS    = RAIZ_PROJETO / "logs"

# Subpastas de reports — cada tipo de saída tem seu próprio diretório
CAMINHO_REPORTS            = RAIZ_PROJETO / "reports"
CAMINHO_REPORTS_EDA        = CAMINHO_REPORTS / "eda"
CAMINHO_REPORTS_TREINO     = CAMINHO_REPORTS / "training"
CAMINHO_REPORTS_AVALIACAO  = CAMINHO_REPORTS / "evaluation"
CAMINHO_REPORTS_GRADCAM    = CAMINHO_REPORTS / "gradcam"

# Garante a existência de todos os diretórios de saída na importação
for _dir in [
    CAMINHO_MODELOS,
    CAMINHO_LOGS,
    CAMINHO_REPORTS_EDA,
    CAMINHO_REPORTS_TREINO,
    CAMINHO_REPORTS_AVALIACAO,
    CAMINHO_REPORTS_GRADCAM,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# MAPEAMENTO DE CLASSES
# ============================================================

# Nomes de subdiretórios no conjunto de teste (nomes curtos)
CLASSES_TESTE = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma",
]

# Nomes de subdiretórios no conjunto de treino/validação (nomes clínicos completos)
CLASSES_TREINO = [
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib",
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
    "normal",
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa",
]

# Mapeia nome do subdiretório → índice inteiro da classe.
# Inclui aliases para os nomes curtos usados no conjunto de teste.
MAPA_CLASSES: dict[str, int] = {
    # Treino / validação
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib":        0,
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa":     1,
    "normal":                                              2,
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa":  3,
    # Aliases — conjunto de teste
    "adenocarcinoma":          0,
    "large.cell.carcinoma":    1,
    "squamous.cell.carcinoma": 3,
}

# Nomes legíveis usados em gráficos e relatórios (ordem = índice da classe)
NOMES_CLASSES: list[str] = [
    "Adenocarcinoma",
    "Carcinoma de Células Grandes",
    "Normal",
    "Carcinoma de Células Escamosas",
]

NUM_CLASSES: int = len(NOMES_CLASSES)

# ============================================================
# HIPERPARÂMETROS DE PRÉ-PROCESSAMENTO
# ============================================================

ALTURA_IMAGEM:  int = 224
LARGURA_IMAGEM: int = 224

# Média e desvio-padrão do ImageNet, usados na normalização para transfer learning
MEDIA_IMAGENET:  list[float] = [0.485, 0.456, 0.406]
DESVIO_IMAGENET: list[float] = [0.229, 0.224, 0.225]

# ============================================================
# HIPERPARÂMETROS DE TREINAMENTO
# ============================================================

TAMANHO_LOTE:      int   = 32    # Amostras por batch
EPOCAS:            int   = 30    # Máximo de épocas
TAXA_APRENDIZADO:  float = 1e-4  # Learning rate inicial (Fase 2 / fine-tuning)
PESO_DECAIMENTO:   float = 1e-4  # Regularização L2
PACIENCIA:         int   = 7     # Épocas sem melhora antes do Early Stopping
SEMENTE_ALEATORIA: int   = 42    # Garante reprodutibilidade dos experimentos

# ============================================================
# CONFIGURAÇÕES DO MODELO
# ============================================================

# Backbone para transfer learning.
# Outras opções válidas: "resnet18", "efficientnet_b0", "densenet121"
ARQUITETURA_MODELO: str = "resnet50"

# Se True, congela o backbone na Fase 1 (treina apenas a cabeça)
CONGELAR_BACKBONE: bool = True

# Nome do checkpoint salvo pelo Early Stopping
NOME_ARQUIVO_MODELO: str = f"melhor_modelo_{ARQUITETURA_MODELO}.pth"
