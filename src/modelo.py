"""
modelo.py
---------
Define a arquitetura do modelo de classificação usando Transfer Learning.
Suporta múltiplos backbones do torchvision com cabeça de classificação customizada.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

from config import NUM_CLASSES, ARQUITETURA_MODELO, CONGELAR_BACKBONE


# ============================================================
# CABEÇA DE CLASSIFICAÇÃO
# ============================================================

class CabecaClassificacao(nn.Module):
    """
    Cabeça de classificação totalmente conectada que substitui
    o classificador original do backbone pré-treinado.

    Parâmetros
    ----------
    num_entradas : int
        Dimensão da saída do backbone (feature vector).
    num_classes : int
        Número de classes de saída.
    taxa_dropout : float
        Probabilidade de dropout para regularização.
    """

    def __init__(
        self,
        num_entradas: int,
        num_classes: int,
        taxa_dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.classificador = nn.Sequential(
            nn.BatchNorm1d(num_entradas),       # Normalização do feature vector
            nn.Dropout(p=taxa_dropout),
            nn.Linear(num_entradas, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=taxa_dropout / 2),
            nn.Linear(256, num_classes),        # Camada de saída
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passagem direta pela cabeça de classificação."""
        return self.classificador(x)


# ============================================================
# MODELO PRINCIPAL
# ============================================================

class ModeloClassificacaoPulmao(nn.Module):
    """
    Modelo de Transfer Learning para classificação de câncer pulmonar.

    Combina um backbone pré-treinado no ImageNet com uma cabeça de
    classificação customizada adaptada para as classes do projeto.

    Parâmetros
    ----------
    arquitetura : str
        Nome do backbone. Opções: 'resnet50', 'resnet18',
        'efficientnet_b0', 'densenet121'.
    num_classes : int
        Número de classes de saída.
    congelar_backbone : bool
        Se True, congela os parâmetros do backbone (útil na fase inicial
        de fine-tuning para treinar apenas a cabeça).
    taxa_dropout : float
        Taxa de dropout na cabeça de classificação.
    """

    ARQUITETURAS_SUPORTADAS = {
        "resnet18":       (models.resnet18,       models.ResNet18_Weights.DEFAULT,       512),
        "resnet50":       (models.resnet50,        models.ResNet50_Weights.DEFAULT,       2048),
        "efficientnet_b0":(models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
        "densenet121":    (models.densenet121,     models.DenseNet121_Weights.DEFAULT,    1024),
    }

    def __init__(
        self,
        arquitetura: str = ARQUITETURA_MODELO,
        num_classes: int = NUM_CLASSES,
        congelar_backbone: bool = CONGELAR_BACKBONE,
        taxa_dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if arquitetura not in self.ARQUITETURAS_SUPORTADAS:
            raise ValueError(
                f"Arquitetura '{arquitetura}' não suportada. "
                f"Escolha entre: {list(self.ARQUITETURAS_SUPORTADAS.keys())}"
            )

        self.arquitetura = arquitetura
        construtor, pesos, dim_features = self.ARQUITETURAS_SUPORTADAS[arquitetura]

        # Carrega backbone pré-treinado no ImageNet
        self.backbone = construtor(weights=pesos)

        if congelar_backbone:
            self._congelar_backbone()

        # Substitui a camada final pelo classificador customizado
        self._substituir_classificador(dim_features, num_classes, taxa_dropout)

    # ----------------------------------------------------------
    # Métodos internos de configuração
    # ----------------------------------------------------------

    def _congelar_backbone(self) -> None:
        """Congela todos os parâmetros do backbone (grad=False)."""
        for parametro in self.backbone.parameters():
            parametro.requires_grad = False

    def descongelar_backbone(self, camadas: Optional[int] = None) -> None:
        """
        Descongela os parâmetros do backbone para fine-tuning completo.

        Parâmetros
        ----------
        camadas : int, optional
            Número de camadas finais a descongelar (contadas do final).
            Se None, descongela todo o backbone.
        """
        todos_parametros = list(self.backbone.parameters())

        if camadas is None:
            for p in todos_parametros:
                p.requires_grad = True
        else:
            for p in todos_parametros[-camadas:]:
                p.requires_grad = True

    def _substituir_classificador(
        self,
        dim_features: int,
        num_classes: int,
        taxa_dropout: float,
    ) -> None:
        """
        Remove a cabeça original do backbone e anexa o classificador customizado.
        Cada arquitetura usa um atributo diferente para a camada final.
        """
        cabeca = CabecaClassificacao(dim_features, num_classes, taxa_dropout)

        if self.arquitetura.startswith("resnet"):
            self.backbone.fc = cabeca

        elif self.arquitetura.startswith("efficientnet"):
            # EfficientNet usa backbone.classifier (Sequential)
            self.backbone.classifier = cabeca

        elif self.arquitetura.startswith("densenet"):
            # DenseNet usa backbone.classifier (Linear simples)
            self.backbone.classifier = cabeca

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem direta completa: backbone → cabeça de classificação.

        Parâmetros
        ----------
        x : torch.Tensor
            Batch de imagens com shape (N, 3, H, W).

        Retorno
        -------
        torch.Tensor
            Logits com shape (N, num_classes). Aplicar softmax para probabilidades.
        """
        return self.backbone(x)

    # ----------------------------------------------------------
    # Utilitários
    # ----------------------------------------------------------

    def contar_parametros(self) -> dict[str, int]:
        """
        Conta parâmetros treináveis e totais do modelo.

        Retorno
        -------
        dict[str, int]
            {'treinavel': ..., 'total': ..., 'congelado': ...}
        """
        total     = sum(p.numel() for p in self.parameters())
        treinavel = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "treinavel": treinavel,
            "total":     total,
            "congelado": total - treinavel,
        }


# ============================================================
# FUNÇÃO DE CONVENIÊNCIA
# ============================================================

def criar_modelo(
    arquitetura: str = ARQUITETURA_MODELO,
    num_classes: int = NUM_CLASSES,
    congelar_backbone: bool = CONGELAR_BACKBONE,
    dispositivo: Optional[torch.device] = None,
) -> ModeloClassificacaoPulmao:
    """
    Cria, configura e move o modelo para o dispositivo correto.

    Parâmetros
    ----------
    arquitetura : str
        Backbone a utilizar.
    num_classes : int
        Número de classes de saída.
    congelar_backbone : bool
        Congela pesos do backbone na inicialização.
    dispositivo : torch.device, optional
        CPU ou GPU. Se None, detecta automaticamente.

    Retorno
    -------
    ModeloClassificacaoPulmao
        Modelo pronto para treinamento.
    """
    if dispositivo is None:
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo = ModeloClassificacaoPulmao(
        arquitetura=arquitetura,
        num_classes=num_classes,
        congelar_backbone=congelar_backbone,
    )
    modelo = modelo.to(dispositivo)

    return modelo
