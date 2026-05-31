"""
avaliacao.py
------------
Funções para avaliar o modelo treinado no conjunto de teste.
Gera métricas, matriz de confusão e visualizações de predição.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from config import NOMES_CLASSES, NUM_CLASSES, CAMINHO_REPORTS


# ============================================================
# COLETA DE PREDIÇÕES
# ============================================================

def coletar_predicoes(
    modelo: nn.Module,
    loader: DataLoader,
    dispositivo: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Roda o modelo sobre todo o DataLoader e coleta predições.

    Parâmetros
    ----------
    modelo : nn.Module
        Modelo treinado.
    loader : DataLoader
        DataLoader do conjunto a avaliar.
    dispositivo : torch.device
        Dispositivo de execução.

    Retorno
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (rotulos_reais, predicoes, probabilidades)
        - rotulos_reais : shape (N,), inteiros
        - predicoes     : shape (N,), inteiros (classe predita)
        - probabilidades: shape (N, num_classes), probabilidades softmax
    """
    modelo.eval()

    rotulos_reais:  List[int]   = []
    predicoes:      List[int]   = []
    probabilidades: List[list]  = []

    with torch.no_grad():
        for imagens, rotulos in tqdm(loader, desc="Avaliando"):
            imagens = imagens.to(dispositivo, non_blocking=True)

            logits  = modelo(imagens)
            probs   = torch.softmax(logits, dim=1).cpu().numpy()
            preds   = logits.argmax(dim=1).cpu().numpy()

            rotulos_reais.extend(rotulos.numpy())
            predicoes.extend(preds)
            probabilidades.extend(probs.tolist())

    return (
        np.array(rotulos_reais),
        np.array(predicoes),
        np.array(probabilidades),
    )


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(
    rotulos_reais: np.ndarray,
    predicoes: np.ndarray,
    probabilidades: np.ndarray,
) -> dict:
    """
    Calcula e imprime as principais métricas de classificação.

    Parâmetros
    ----------
    rotulos_reais : np.ndarray
        Rótulos verdadeiros.
    predicoes : np.ndarray
        Predições do modelo.
    probabilidades : np.ndarray
        Probabilidades softmax por classe.

    Retorno
    -------
    dict
        Dicionário com acurácia, AUC-ROC e relatório de classificação.
    """
    acuracia = (rotulos_reais == predicoes).mean()

    # AUC-ROC multiclasse (estratégia One-vs-Rest)
    auc_roc = roc_auc_score(
        rotulos_reais,
        probabilidades,
        multi_class="ovr",
        average="macro",
    )

    relatorio = classification_report(
        rotulos_reais,
        predicoes,
        target_names=NOMES_CLASSES,
        digits=4,
    )

    print(f"\n{'=' * 55}")
    print(f"  RESULTADOS NO CONJUNTO DE TESTE")
    print(f"{'=' * 55}")
    print(f"  Acurácia : {acuracia:.4f} ({acuracia*100:.2f}%)")
    print(f"  AUC-ROC  : {auc_roc:.4f}")
    print(f"\n{relatorio}")

    return {
        "acuracia":  acuracia,
        "auc_roc":   auc_roc,
        "relatorio": relatorio,
    }


# ============================================================
# VISUALIZAÇÕES
# ============================================================

def plotar_matriz_confusao(
    rotulos_reais: np.ndarray,
    predicoes: np.ndarray,
    salvar: bool = True,
    nome_arquivo: str = "matriz_confusao.png",
) -> None:
    """
    Plota a matriz de confusão normalizada e absoluta lado a lado.

    Parâmetros
    ----------
    rotulos_reais : np.ndarray
        Rótulos verdadeiros.
    predicoes : np.ndarray
        Predições do modelo.
    salvar : bool
        Se True, salva o gráfico em CAMINHO_REPORTS.
    nome_arquivo : str
        Nome do arquivo PNG de saída.
    """
    mc          = confusion_matrix(rotulos_reais, predicoes)
    mc_norm     = mc.astype(float) / mc.sum(axis=1, keepdims=True)

    nomes_curtos = ["Adeno.", "Gr.Cell.", "Normal", "Sq.Cell."]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, dados, titulo, fmt in zip(
        axes,
        [mc, mc_norm],
        ["Contagens Absolutas", "Normalizada por Linha"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            dados,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=nomes_curtos,
            yticklabels=nomes_curtos,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(f"Matriz de Confusão — {titulo}", fontsize=13, pad=12)
        ax.set_xlabel("Predito", fontsize=11)
        ax.set_ylabel("Real", fontsize=11)

    plt.suptitle("Avaliação do Modelo — Conjunto de Teste", fontsize=14, y=1.02)
    plt.tight_layout()

    if salvar:
        caminho = CAMINHO_REPORTS / nome_arquivo
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        print(f"Matriz de confusão salva em: {caminho}")

    plt.show()


def plotar_curvas_roc(
    rotulos_reais: np.ndarray,
    probabilidades: np.ndarray,
    salvar: bool = True,
    nome_arquivo: str = "curvas_roc.png",
) -> None:
    """
    Plota as curvas ROC individuais por classe (estratégia OvR).

    Parâmetros
    ----------
    rotulos_reais : np.ndarray
        Rótulos verdadeiros.
    probabilidades : np.ndarray
        Probabilidades softmax, shape (N, num_classes).
    salvar : bool
        Se True, salva o gráfico em CAMINHO_REPORTS.
    nome_arquivo : str
        Nome do arquivo PNG de saída.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cores = plt.cm.tab10(np.linspace(0, 0.5, NUM_CLASSES))

    for idx, (nome, cor) in enumerate(zip(NOMES_CLASSES, cores)):
        # Binariza os rótulos para a classe atual (OvR)
        rotulos_bin = (rotulos_reais == idx).astype(int)
        fpr, tpr, _ = roc_curve(rotulos_bin, probabilidades[:, idx])
        auc         = roc_auc_score(rotulos_bin, probabilidades[:, idx])

        ax.plot(fpr, tpr, color=cor, lw=2, label=f"{nome} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiros Positivos", fontsize=12)
    ax.set_title("Curvas ROC por Classe (One-vs-Rest)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if salvar:
        caminho = CAMINHO_REPORTS / nome_arquivo
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        print(f"Curvas ROC salvas em: {caminho}")

    plt.show()


def plotar_historico_treino(
    historico: dict,
    salvar: bool = True,
    nome_arquivo: str = "historico_treino.png",
) -> None:
    """
    Plota as curvas de loss e acurácia ao longo das épocas.

    Parâmetros
    ----------
    historico : dict
        Dicionário retornado por treinar_modelo().
    salvar : bool
        Se True, salva o gráfico em CAMINHO_REPORTS.
    nome_arquivo : str
        Nome do arquivo PNG de saída.
    """
    epocas = range(1, len(historico["loss_treino"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Loss ---
    ax1.plot(epocas, historico["loss_treino"],    "b-o", ms=4, label="Treino")
    ax1.plot(epocas, historico["loss_validacao"], "r-o", ms=4, label="Validação")
    ax1.set_title("Loss por Época", fontsize=13)
    ax1.set_xlabel("Época", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # --- Acurácia ---
    ax2.plot(epocas, historico["acuracia_treino"],    "b-o", ms=4, label="Treino")
    ax2.plot(epocas, historico["acuracia_validacao"], "r-o", ms=4, label="Validação")
    ax2.set_title("Acurácia por Época", fontsize=13)
    ax2.set_xlabel("Época", fontsize=11)
    ax2.set_ylabel("Acurácia", fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.suptitle("Histórico de Treinamento", fontsize=14)
    plt.tight_layout()

    if salvar:
        caminho = CAMINHO_REPORTS / nome_arquivo
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        print(f"Histórico salvo em: {caminho}")

    plt.show()
