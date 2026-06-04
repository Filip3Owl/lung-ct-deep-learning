"""
modelo.py
---------
Arquitetura do modelo de classificação de câncer pulmonar via Transfer Learning.

O modelo combina um backbone pré-treinado no ImageNet com uma cabeça de
classificação customizada. A estratégia de duas fases permite convergência
estável mesmo com poucos dados:

  Fase 1 — Backbone congelado
      Apenas a cabeça é treinada (congelar_backbone=True). Permite que o
      classificador se adapte ao domínio médico sem destruir os pesos do
      ImageNet com gradientes ruidosos na inicialização.

  Fase 2 — Fine-tuning
      As últimas N camadas do backbone são descongeladas com taxa de
      aprendizado baixa, refinando as features de alto nível para TC pulmonar.

Exportações principais
----------------------
CabecaClassificacao          Módulo nn com a cabeça totalmente conectada
ModeloClassificacaoPulmao    Modelo completo (backbone + cabeça)
criar_modelo(...)            Função de conveniência: cria e move para dispositivo

Uso
---
    from modelo import criar_modelo

    # Fase 1 — backbone congelado
    modelo = criar_modelo(congelar_backbone=True)

    # Fase 2 — descongelamento das últimas 30 camadas
    modelo.descongelar_backbone(camadas=30)
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
    Cabeça totalmente conectada que substitui o classificador original do backbone.

    Arquitetura
    -----------
    BatchNorm1d(dim_entrada)
        → Dropout(taxa_dropout)
        → Linear(dim_entrada → 256)
        → ReLU
        → BatchNorm1d(256)
        → Dropout(taxa_dropout / 2)
        → Linear(256 → num_classes)   ← logits de saída

    O BatchNorm na entrada normaliza o feature vector do backbone, acelerando
    a convergência da cabeça durante a Fase 1. O Dropout duplo regulariza
    o classificador para datasets pequenos.

    Parâmetros
    ----------
    num_entradas : int
        Dimensão do feature vector de saída do backbone.
        ResNet50 → 2048 | ResNet18 → 512 | EfficientNet-B0 → 1280 | DenseNet121 → 1024
    num_classes : int
        Número de classes de saída (logits, sem softmax).
    taxa_dropout : float
        Probabilidade de zeragem na primeira camada Dropout.
    """

    def __init__(
        self,
        num_entradas:  int,
        num_classes:   int,
        taxa_dropout:  float = 0.5,
    ) -> None:
        super().__init__()

        self.classificador = nn.Sequential(
            nn.BatchNorm1d(num_entradas),
            nn.Dropout(p=taxa_dropout),
            nn.Linear(num_entradas, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=taxa_dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parâmetros
        ----------
        x : torch.Tensor
            Feature vector com shape (N, num_entradas).

        Retorno
        -------
        torch.Tensor
            Logits com shape (N, num_classes).
        """
        return self.classificador(x)


# ============================================================
# MODELO PRINCIPAL
# ============================================================

class ModeloClassificacaoPulmao(nn.Module):
    """
    Modelo de Transfer Learning para classificação de TC pulmonar em 4 classes.

    Carrega um backbone pré-treinado no ImageNet, remove sua cabeça original
    e substitui por CabecaClassificacao adaptada ao número de classes do projeto.

    Backbones suportados
    --------------------
    'resnet50'       → 2048 features  (padrão)
    'resnet18'       → 512  features
    'efficientnet_b0'→ 1280 features
    'densenet121'    → 1024 features

    Parâmetros
    ----------
    arquitetura : str
        Nome do backbone. Ver ARQUITETURAS_SUPORTADAS.
    num_classes : int
        Número de classes de saída.
    congelar_backbone : bool
        Se True, congela todos os pesos do backbone na inicialização.
    taxa_dropout : float
        Taxa de dropout na CabecaClassificacao.

    Raises
    ------
    ValueError
        Se `arquitetura` não estiver em ARQUITETURAS_SUPORTADAS.
    """

    ARQUITETURAS_SUPORTADAS: dict[str, tuple] = {
        "resnet18":        (models.resnet18,        models.ResNet18_Weights.DEFAULT,        512),
        "resnet50":        (models.resnet50,         models.ResNet50_Weights.DEFAULT,        2048),
        "efficientnet_b0": (models.efficientnet_b0,  models.EfficientNet_B0_Weights.DEFAULT, 1280),
        "densenet121":     (models.densenet121,      models.DenseNet121_Weights.DEFAULT,     1024),
    }

    def __init__(
        self,
        arquitetura:       str   = ARQUITETURA_MODELO,
        num_classes:       int   = NUM_CLASSES,
        congelar_backbone: bool  = CONGELAR_BACKBONE,
        taxa_dropout:      float = 0.5,
    ) -> None:
        super().__init__()

        if arquitetura not in self.ARQUITETURAS_SUPORTADAS:
            opcoes = list(self.ARQUITETURAS_SUPORTADAS.keys())
            raise ValueError(f"Arquitetura '{arquitetura}' não suportada. Opções: {opcoes}")

        self.arquitetura = arquitetura
        construtor, pesos, dim_features = self.ARQUITETURAS_SUPORTADAS[arquitetura]

        self.backbone = construtor(weights=pesos)

        if congelar_backbone:
            self._congelar_backbone()

        self._substituir_classificador(dim_features, num_classes, taxa_dropout)

    # ----------------------------------------------------------
    # Configuração do backbone
    # ----------------------------------------------------------

    def _congelar_backbone(self) -> None:
        """Congela todos os parâmetros do backbone (requires_grad = False)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def descongelar_backbone(self, camadas: Optional[int] = None) -> None:
        """
        Descongela parâmetros do backbone para a Fase 2 de fine-tuning.

        Parâmetros
        ----------
        camadas : int, optional
            Número de parâmetros finais a descongelar (contados do último).
            None descongela todo o backbone.

        Exemplo
        -------
            modelo.descongelar_backbone(camadas=30)  # Fase 2 típica no ResNet50
        """
        todos = list(self.backbone.parameters())
        alvo  = todos if camadas is None else todos[-camadas:]
        for p in alvo:
            p.requires_grad = True

    def _substituir_classificador(
        self,
        dim_features: int,
        num_classes:  int,
        taxa_dropout: float,
    ) -> None:
        """
        Remove a cabeça original do backbone e anexa CabecaClassificacao.
        Cada família de arquitetura expõe a cabeça em um atributo diferente.
        """
        cabeca = CabecaClassificacao(dim_features, num_classes, taxa_dropout)

        if self.arquitetura.startswith("resnet"):
            self.backbone.fc = cabeca
        elif self.arquitetura.startswith(("efficientnet", "densenet")):
            self.backbone.classifier = cabeca

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem direta: backbone → CabecaClassificacao → logits.

        Parâmetros
        ----------
        x : torch.Tensor
            Batch de imagens com shape (N, 3, H, W).

        Retorno
        -------
        torch.Tensor
            Logits com shape (N, num_classes). Use softmax para probabilidades.
        """
        return self.backbone(x)

    # ----------------------------------------------------------
    # Utilitários de inspeção
    # ----------------------------------------------------------

    def contar_parametros(self) -> dict[str, int]:
        """
        Contagem de parâmetros treináveis, congelados e totais.

        Retorno
        -------
        dict[str, int]
            {'treinavel': ..., 'congelado': ..., 'total': ...}
        """
        total     = sum(p.numel() for p in self.parameters())
        treinavel = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "treinavel": treinavel,
            "congelado": total - treinavel,
            "total":     total,
        }


# ============================================================
# FUNÇÃO DE CONVENIÊNCIA
# ============================================================

def criar_modelo(
    arquitetura:       str                    = ARQUITETURA_MODELO,
    num_classes:       int                    = NUM_CLASSES,
    congelar_backbone: bool                   = CONGELAR_BACKBONE,
    dispositivo:       Optional[torch.device] = None,
) -> ModeloClassificacaoPulmao:
    """
    Cria, configura e move o modelo para o dispositivo correto.

    Detecta automaticamente GPU (CUDA) ou CPU se `dispositivo` não for fornecido.

    Parâmetros
    ----------
    arquitetura : str
        Backbone a utilizar.
    num_classes : int
        Número de classes de saída.
    congelar_backbone : bool
        Congela o backbone na inicialização (recomendado para Fase 1).
    dispositivo : torch.device, optional
        Dispositivo de destino. Detectado automaticamente se None.

    Retorno
    -------
    ModeloClassificacaoPulmao
        Modelo pronto para treinamento no dispositivo selecionado.

    Exemplo
    -------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modelo = criar_modelo(congelar_backbone=True, dispositivo=device)
        params = modelo.contar_parametros()
        print(f"Parâmetros treináveis: {params['treinavel']:,}")
    """
    if dispositivo is None:
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo = ModeloClassificacaoPulmao(
        arquitetura=arquitetura,
        num_classes=num_classes,
        congelar_backbone=congelar_backbone,
    )
    return modelo.to(dispositivo)
