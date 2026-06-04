"""
avaliacao.py
------------
Avaliação do modelo treinado e geração de visualizações de resultados.

Responsabilidades
-----------------
- Coletar predições e probabilidades softmax em qualquer DataLoader
- Calcular e imprimir métricas (acurácia, AUC-ROC, classification report)
- Gerar e salvar gráficos prontos para relatório:
    · Matriz de confusão (absoluta + normalizada)
    · Curvas ROC por classe (One-vs-Rest)
    · Histórico de treinamento (loss e acurácia por época)

Saídas salvas automaticamente
------------------------------
reports/evaluation/matriz_confusao.png
reports/evaluation/curvas_roc.png
reports/training/historico_treino.png

Exportações principais
----------------------
coletar_predicoes(...)        Roda o modelo e retorna arrays de predição
calcular_metricas(...)        Calcula e imprime métricas no conjunto de teste
plotar_matriz_confusao(...)   Heatmap de confusão duplo (absoluto + normalizado)
plotar_curvas_roc(...)        Curvas ROC individuais por classe
plotar_historico_treino(...)  Curvas de loss e acurácia do histórico de treino

Uso
---
    from avaliacao import coletar_predicoes, calcular_metricas, plotar_matriz_confusao

    rotulos, predicoes, probs = coletar_predicoes(modelo, loader_teste, dispositivo)
    calcular_metricas(rotulos, predicoes, probs)
    plotar_matriz_confusao(rotulos, predicoes)
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from config import (
    NOMES_CLASSES,
    NUM_CLASSES,
    CAMINHO_REPORTS_AVALIACAO,
    CAMINHO_REPORTS_TREINO,
)


# ============================================================
# COLETA DE PREDIÇÕES
# ============================================================

def coletar_predicoes(
    modelo:      nn.Module,
    loader:      DataLoader,
    dispositivo: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Passa o DataLoader inteiro pelo modelo e coleta predições.

    Executa em modo de avaliação (model.eval() + torch.no_grad()) para
    garantir comportamento determinístico e economia de memória.

    Parâmetros
    ----------
    modelo : nn.Module
        Modelo treinado.
    loader : DataLoader
        DataLoader do conjunto a avaliar (validação ou teste).
    dispositivo : torch.device
        Dispositivo onde o modelo está alocado.

    Retorno
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        rotulos_reais   — shape (N,), inteiros, rótulos verdadeiros
        predicoes       — shape (N,), inteiros, classe predita (argmax)
        probabilidades  — shape (N, num_classes), probabilidades softmax
    """
    modelo.eval()

    rotulos_reais:  List[int]  = []
    predicoes:      List[int]  = []
    probabilidades: List[list] = []

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
    rotulos_reais:  np.ndarray,
    predicoes:      np.ndarray,
    probabilidades: np.ndarray,
) -> dict:
    """
    Calcula e imprime as principais métricas de classificação multiclasse.

    Métricas calculadas
    -------------------
    Acurácia global
        Proporção de predições corretas sobre o total de amostras.
    AUC-ROC macro (One-vs-Rest)
        Média das áreas sob a curva ROC para cada classe tratada como
        binária. Robusta ao desbalanceamento de classes.
    Classification report (sklearn)
        Precisão, recall e F1-score por classe, além de médias macro e
        ponderada.

    Parâmetros
    ----------
    rotulos_reais : np.ndarray
        Rótulos verdadeiros, shape (N,).
    predicoes : np.ndarray
        Predições do modelo, shape (N,).
    probabilidades : np.ndarray
        Probabilidades softmax, shape (N, num_classes).

    Retorno
    -------
    dict
        {'acuracia': float, 'auc_roc': float, 'relatorio': str}
    """
    acuracia = (rotulos_reais == predicoes).mean()

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
    print(f"  Acurácia : {acuracia:.4f} ({acuracia * 100:.2f}%)")
    print(f"  AUC-ROC  : {auc_roc:.4f}")
    print(f"\n{relatorio}")

    return {"acuracia": acuracia, "auc_roc": auc_roc, "relatorio": relatorio}


# ============================================================
# VISUALIZAÇÕES
# ============================================================

def plotar_matriz_confusao(
    rotulos_reais: np.ndarray,
    predicoes:     np.ndarray,
    salvar:        bool = True,
    nome_arquivo:  str  = "matriz_confusao.png",
) -> None:
    """
    Plota a matriz de confusão em contagens absolutas e normalizada por linha.

    Apresenta duas visualizações lado a lado para facilitar a interpretação:
    a versão absoluta mostra quantas amostras foram confundidas; a normalizada
    revela a taxa de erro por classe independentemente do tamanho do grupo.

    Parâmetros
    ----------
    rotulos_reais : np.ndarray
        Rótulos verdadeiros, shape (N,).
    predicoes : np.ndarray
        Predições do modelo, shape (N,).
    salvar : bool
        Se True, salva o gráfico em reports/evaluation/.
    nome_arquivo : str
        Nome do arquivo PNG de saída.
    """
    mc      = confusion_matrix(rotulos_reais, predicoes)
    mc_norm = mc.astype(float) / mc.sum(axis=1, keepdims=True)

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
            annot=True, fmt=fmt, cmap="Blues",
            xticklabels=nomes_curtos,
            yticklabels=nomes_curtos,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Matriz de Confusão — {titulo}", fontsize=13, pad=12)
        ax.set_xlabel("Predito",  fontsize=11)
        ax.set_ylabel("Real",     fontsize=11)

    plt.suptitle("Avaliação do Modelo — Conjunto de Teste", fontsize=14, y=1.02)
    plt.tight_layout()

    if salvar:
        caminho = CAMINHO_REPORTS_AVALIACAO / nome_arquivo
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        print(f"Matriz de confusão salva em: {caminho}")

    plt.show()


def plotar_curvas_roc(
    rotulos_reais:  np.ndarray,
    probabilidades: np.ndarray,
    salvar:         bool = True,
    nome_arquivo:   str  = "curvas_roc.png",
) -> None:
    """
    Plota as curvas ROC individuais por classe usando a estratégia One-vs-Rest.

    Cada classe é tratada como um problema binário: a curva mostra a troca
    entre taxa de verdadeiros positivos e falsos positivos ao variar o limiar
    de decisão. A área sob a curva (AUC) de cada classe é exibida na legenda.

    Parâmetros
    ----------
    rotulos_reais : np.ndarray
        Rótulos verdadeiros, shape (N,).
    probabilidades : np.ndarray
        Probabilidades softmax, shape (N, num_classes).
    salvar : bool
        Se True, salva o gráfico em reports/evaluation/.
    nome_arquivo : str
        Nome do arquivo PNG de saída.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cores = plt.cm.tab10(np.linspace(0, 0.5, NUM_CLASSES))

    for idx, (nome, cor) in enumerate(zip(NOMES_CLASSES, cores)):
        rotulos_bin = (rotulos_reais == idx).astype(int)
        fpr, tpr, _ = roc_curve(rotulos_bin, probabilidades[:, idx])
        auc         = roc_auc_score(rotulos_bin, probabilidades[:, idx])
        ax.plot(fpr, tpr, color=cor, lw=2, label=f"{nome} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos",      fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiros Positivos", fontsize=12)
    ax.set_title("Curvas ROC por Classe (One-vs-Rest)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if salvar:
        caminho = CAMINHO_REPORTS_AVALIACAO / nome_arquivo
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        print(f"Curvas ROC salvas em: {caminho}")

    plt.show()


def plotar_historico_treino(
    historico:    dict,
    salvar:       bool = True,
    nome_arquivo: str  = "historico_treino.png",
) -> None:
    """
    Plota as curvas de loss e acurácia ao longo das épocas de treinamento.

    Exibe treino e validação lado a lado para facilitar a identificação de
    overfitting (divergência entre as curvas) e da época de melhor generalização.

    Parâmetros
    ----------
    historico : dict
        Dicionário retornado por treinar_modelo(), com as chaves:
        'loss_treino', 'loss_validacao', 'acuracia_treino', 'acuracia_validacao'.
    salvar : bool
        Se True, salva o gráfico em reports/training/.
    nome_arquivo : str
        Nome do arquivo PNG de saída.
    """
    epocas = range(1, len(historico["loss_treino"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epocas, historico["loss_treino"],    "b-o", ms=4, label="Treino")
    ax1.plot(epocas, historico["loss_validacao"], "r-o", ms=4, label="Validação")
    ax1.set_title("Loss por Época",        fontsize=13)
    ax1.set_xlabel("Época",                fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss",   fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2.plot(epocas, historico["acuracia_treino"],    "b-o", ms=4, label="Treino")
    ax2.plot(epocas, historico["acuracia_validacao"], "r-o", ms=4, label="Validação")
    ax2.set_title("Acurácia por Época", fontsize=13)
    ax2.set_xlabel("Época",             fontsize=11)
    ax2.set_ylabel("Acurácia",          fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.suptitle("Histórico de Treinamento", fontsize=14)
    plt.tight_layout()

    if salvar:
        caminho = CAMINHO_REPORTS_TREINO / nome_arquivo
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        print(f"Histórico salvo em: {caminho}")

    plt.show()
