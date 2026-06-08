"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping) para explicabilidade
do modelo de classificação de câncer pulmonar.

Grad-CAM captura os gradientes da classe de interesse em relação aos mapas de
ativação da última camada convolucional. A média global desses gradientes
pondera cada mapa de ativação; a soma ponderada — passada por ReLU — indica
quais regiões da imagem mais contribuíram para a predição.

Referência: Selvaraju et al. (2017) — https://arxiv.org/abs/1610.02391

Exportações principais
----------------------
GradCAM                       Classe que registra hooks e computa o mapa de calor
visualizar_gradcam(...)       Plota grade de imagens com heatmap sobreposto
inferencia_com_gradcam(...)   Carrega uma imagem, prediz a classe e exibe Grad-CAM

Uso rápido
----------
    from gradcam import inferencia_com_gradcam
    from modelo import criar_modelo
    import torch

    device  = torch.device("cpu")
    modelo  = criar_modelo(congelar_backbone=False, dispositivo=device)
    modelo.load_state_dict(torch.load("modelos_salvos/melhor_modelo_resnet50.pth", map_location=device))

    inferencia_com_gradcam(modelo, "Data/test/adenocarcinoma/000001.png", device)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from config import (
    NOMES_CLASSES,
    ALTURA_IMAGEM,
    LARGURA_IMAGEM,
    MEDIA_IMAGENET,
    DESVIO_IMAGENET,
    CAMINHO_REPORTS_GRADCAM,
)


# ============================================================
# CLASSE GRAD-CAM
# ============================================================

class GradCAM:
    """
    Computa mapas de ativação Grad-CAM para modelos ResNet.

    Registra hooks de forward (ativações) e backward (gradientes) na camada
    alvo. Após chamar `computar()`, os hooks são removidos automaticamente.

    Parâmetros
    ----------
    modelo : nn.Module
        Modelo treinado. Deve expor `backbone.layer4` (ResNet).
    camada_alvo : nn.Module, optional
        Camada convolucional alvo. Se None, usa `modelo.backbone.layer4`.
    """

    def __init__(self, modelo: nn.Module, camada_alvo: Optional[nn.Module] = None) -> None:
        self.modelo = modelo
        self.camada_alvo = camada_alvo or modelo.backbone.layer4

        self._ativacoes: Optional[torch.Tensor] = None
        self._gradientes: Optional[torch.Tensor] = None
        self._hooks: list = []

        self._registrar_hooks()

    def _registrar_hooks(self) -> None:
        self._hooks.append(
            self.camada_alvo.register_forward_hook(self._hook_ativacao)
        )
        self._hooks.append(
            self.camada_alvo.register_full_backward_hook(self._hook_gradiente)
        )

    def _hook_ativacao(self, modulo, entrada, saida) -> None:
        self._ativacoes = saida.detach()

    def _hook_gradiente(self, modulo, grad_entrada, grad_saida) -> None:
        self._gradientes = grad_saida[0].detach()

    def remover_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def computar(
        self,
        tensor_imagem: torch.Tensor,
        classe_alvo: Optional[int] = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Executa forward + backward e retorna o mapa de calor normalizado.

        Parâmetros
        ----------
        tensor_imagem : torch.Tensor
            Imagem pré-processada, shape (1, 3, H, W).
        classe_alvo : int, optional
            Classe para a qual calcular o Grad-CAM. Se None, usa a predita.

        Retorno
        -------
        tuple[np.ndarray, int, float]
            mapa_calor   — shape (H, W), valores em [0, 1]
            classe_idx   — índice da classe usada no backward
            confianca    — probabilidade softmax da classe predita
        """
        self.modelo.eval()
        tensor_imagem = tensor_imagem.requires_grad_(False)

        logits = self.modelo(tensor_imagem)
        probs  = torch.softmax(logits, dim=1)

        if classe_alvo is None:
            classe_alvo = logits.argmax(dim=1).item()

        confianca = probs[0, classe_alvo].item()

        self.modelo.zero_grad()
        logits[0, classe_alvo].backward()

        # Média global dos gradientes por canal → pesos α
        pesos = self._gradientes.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Combinação linear ponderada dos mapas de ativação
        mapa = (pesos * self._ativacoes).sum(dim=1).squeeze()    # (H', W')
        mapa = torch.relu(mapa).cpu().numpy()

        # Normaliza para [0, 1]
        if mapa.max() > 0:
            mapa = mapa / mapa.max()

        return mapa, classe_alvo, confianca

    def __del__(self) -> None:
        self.remover_hooks()


# ============================================================
# VISUALIZAÇÃO
# ============================================================

def _sobrepor_heatmap(imagem_orig: np.ndarray, mapa: np.ndarray) -> np.ndarray:
    """Redimensiona o mapa de calor e o sobrepõe à imagem original."""
    h, w = imagem_orig.shape[:2]
    mapa_pil = Image.fromarray(np.uint8(mapa * 255)).resize((w, h), Image.BILINEAR)
    mapa_np  = np.array(mapa_pil) / 255.0

    heatmap  = cm.jet(mapa_np)[:, :, :3]          # (H, W, 3) RGB
    overlay  = 0.55 * imagem_orig + 0.45 * heatmap
    return np.clip(overlay, 0, 1)


def visualizar_gradcam(
    modelo:         nn.Module,
    imagens:        torch.Tensor,
    rotulos_reais:  list[int],
    dispositivo:    torch.device,
    n_imagens:      int  = 8,
    salvar:         bool = True,
    nome_arquivo:   str  = "gradcam_grid.png",
) -> None:
    """
    Plota uma grade com as imagens originais e seus mapas Grad-CAM sobrepostos.

    Parâmetros
    ----------
    modelo : nn.Module
        Modelo treinado.
    imagens : torch.Tensor
        Batch de imagens pré-processadas, shape (N, 3, H, W).
    rotulos_reais : list[int]
        Rótulos verdadeiros de cada imagem no batch.
    dispositivo : torch.device
        Dispositivo onde o modelo está alocado.
    n_imagens : int
        Número de imagens a exibir (máximo = tamanho do batch).
    salvar : bool
        Se True, salva a figura em reports/gradcam/.
    nome_arquivo : str
        Nome do arquivo PNG de saída.
    """
    n = min(n_imagens, len(imagens))
    fig, eixos = plt.subplots(2, n, figsize=(3 * n, 6))

    mean = torch.tensor(MEDIA_IMAGENET).view(3, 1, 1)
    std  = torch.tensor(DESVIO_IMAGENET).view(3, 1, 1)

    for i in range(n):
        tensor = imagens[i].unsqueeze(0).to(dispositivo)
        gradcam = GradCAM(modelo)
        mapa, classe_pred, confianca = gradcam.computar(tensor)
        gradcam.remover_hooks()

        # Desnormaliza para visualização
        img_np = (imagens[i].cpu() * std + mean).permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        overlay = _sobrepor_heatmap(img_np, mapa)

        rotulo_real = NOMES_CLASSES[rotulos_reais[i]]
        rotulo_pred = NOMES_CLASSES[classe_pred]
        cor_titulo  = "green" if classe_pred == rotulos_reais[i] else "red"

        eixos[0, i].imshow(img_np)
        eixos[0, i].set_title(f"Real: {rotulo_real}", fontsize=8)
        eixos[0, i].axis("off")

        eixos[1, i].imshow(overlay)
        eixos[1, i].set_title(
            f"Pred: {rotulo_pred}\n({confianca:.0%})",
            fontsize=8, color=cor_titulo,
        )
        eixos[1, i].axis("off")

    plt.suptitle("Grad-CAM — Linha superior: original | Linha inferior: mapa de atenção", fontsize=11)
    plt.tight_layout()

    if salvar:
        caminho = CAMINHO_REPORTS_GRADCAM / nome_arquivo
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM salvo em: {caminho}")

    plt.show()


# ============================================================
# INFERÊNCIA EM IMAGEM AVULSA
# ============================================================

_TRANSFORM_INFERENCIA = T.Compose([
    T.Resize((ALTURA_IMAGEM, LARGURA_IMAGEM)),
    T.ToTensor(),
    T.Normalize(mean=MEDIA_IMAGENET, std=DESVIO_IMAGENET),
])


def inferencia_com_gradcam(
    modelo:        nn.Module,
    caminho_imagem: str | Path,
    dispositivo:   torch.device,
    salvar:        bool = True,
) -> dict:
    """
    Carrega uma imagem, prediz a classe e exibe o mapa Grad-CAM.

    Parâmetros
    ----------
    modelo : nn.Module
        Modelo treinado.
    caminho_imagem : str | Path
        Caminho para a imagem de entrada (PNG/JPG).
    dispositivo : torch.device
        Dispositivo onde o modelo está alocado.
    salvar : bool
        Se True, salva a figura em reports/gradcam/.

    Retorno
    -------
    dict
        {'classe': str, 'confianca': float, 'probabilidades': dict}
    """
    caminho_imagem = Path(caminho_imagem)
    imagem_pil = Image.open(caminho_imagem).convert("RGB")
    tensor     = _TRANSFORM_INFERENCIA(imagem_pil).unsqueeze(0).to(dispositivo)

    gradcam = GradCAM(modelo)
    mapa, classe_idx, confianca = gradcam.computar(tensor)
    gradcam.remover_hooks()

    # Probabilidades de todas as classes
    with torch.no_grad():
        logits = modelo(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    mean = torch.tensor(MEDIA_IMAGENET).view(3, 1, 1)
    std  = torch.tensor(DESVIO_IMAGENET).view(3, 1, 1)
    img_np = (tensor.squeeze().cpu() * std + mean).permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    overlay = _sobrepor_heatmap(img_np, mapa)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow(img_np)
    ax1.set_title("Imagem original", fontsize=12)
    ax1.axis("off")

    ax2.imshow(overlay)
    ax2.set_title(
        f"Grad-CAM — {NOMES_CLASSES[classe_idx]}\nConfiança: {confianca:.1%}",
        fontsize=12,
    )
    ax2.axis("off")

    # Barra de probabilidades
    prob_texto = "\n".join(
        f"{NOMES_CLASSES[i]}: {probs[i]:.1%}" for i in range(len(NOMES_CLASSES))
    )
    fig.text(
        0.5, -0.05, prob_texto,
        ha="center", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.suptitle(f"Inferência: {caminho_imagem.name}", fontsize=13)
    plt.tight_layout()

    if salvar:
        nome_saida = f"gradcam_{caminho_imagem.stem}.png"
        caminho_saida = CAMINHO_REPORTS_GRADCAM / nome_saida
        plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
        print(f"Resultado salvo em: {caminho_saida}")

    plt.show()

    return {
        "classe":         NOMES_CLASSES[classe_idx],
        "confianca":      confianca,
        "probabilidades": {NOMES_CLASSES[i]: float(probs[i]) for i in range(len(NOMES_CLASSES))},
    }
