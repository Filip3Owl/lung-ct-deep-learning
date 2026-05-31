"""
dataset.py
----------
Define o Dataset customizado PyTorch para as imagens de tomografia pulmonar.
Responsável por carregar, transformar e fornecer as imagens ao DataLoader.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import (
    CAMINHO_TREINO, CAMINHO_TESTE, CAMINHO_VALIDACAO,
    ALTURA_IMAGEM, LARGURA_IMAGEM,
    MEDIA_IMAGENET, DESVIO_IMAGENET,
    MAPA_CLASSES, NUM_CLASSES, NOMES_CLASSES,
    TAMANHO_LOTE, SEMENTE_ALEATORIA,
)


# ============================================================
# TRANSFORMAÇÕES (augmentações de dados)
# ============================================================

def obter_transformacoes(modo: str) -> T.Compose:
    """
    Retorna o pipeline de transformações de imagem para cada modo.

    Parâmetros
    ----------
    modo : str
        Um de 'treino', 'validacao' ou 'teste'.

    Retorno
    -------
    torchvision.transforms.Compose
        Pipeline de transformações.

    Notas
    -----
    - No modo 'treino' aplicamos augmentações aleatórias para regularização.
    - Nos modos 'validacao' e 'teste' apenas normalizamos, sem aleatoriedade.
    """
    if modo == "treino":
        return T.Compose([
            T.Resize((ALTURA_IMAGEM, LARGURA_IMAGEM)),
            T.RandomHorizontalFlip(p=0.5),              # Espelhamento horizontal
            T.RandomVerticalFlip(p=0.2),                # Espelhamento vertical ocasional
            T.RandomRotation(degrees=15),               # Rotação de até ±15°
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Translação leve
            T.ToTensor(),
            T.Normalize(mean=MEDIA_IMAGENET, std=DESVIO_IMAGENET),
        ])
    else:
        return T.Compose([
            T.Resize((ALTURA_IMAGEM, LARGURA_IMAGEM)),
            T.ToTensor(),
            T.Normalize(mean=MEDIA_IMAGENET, std=DESVIO_IMAGENET),
        ])


# ============================================================
# DATASET PYTORCH
# ============================================================

class DatasetPulmao(Dataset):
    """
    Dataset PyTorch para classificação de imagens de tomografia pulmonar.

    A estrutura de diretórios esperada é:
        raiz/
            <classe_a>/
                imagem1.png
                imagem2.png
            <classe_b>/
                ...

    Parâmetros
    ----------
    caminho_raiz : str | Path
        Caminho para o diretório raiz do conjunto de dados.
    transformacoes : callable, optional
        Pipeline de transformações a aplicar em cada imagem.
    extensoes_validas : tuple[str], optional
        Extensões de arquivo aceitas como imagem.
    """

    EXTENSOES_VALIDAS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")

    def __init__(
        self,
        caminho_raiz: Path | str,
        transformacoes: Optional[Callable] = None,
        extensoes_validas: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.caminho_raiz = Path(caminho_raiz)
        self.transformacoes = transformacoes
        self.extensoes_validas = extensoes_validas or self.EXTENSOES_VALIDAS

        # Listas paralelas: caminho da imagem e rótulo inteiro correspondente
        self.caminhos_imagens: list[Path] = []
        self.rotulos: list[int] = []

        self._carregar_dataset()

    def _carregar_dataset(self) -> None:
        """
        Percorre os subdiretórios de caminho_raiz e popula
        self.caminhos_imagens e self.rotulos.
        Ignora subdiretórios cujo nome não esteja em MAPA_CLASSES.
        """
        if not self.caminho_raiz.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {self.caminho_raiz}")

        for subdir in sorted(self.caminho_raiz.iterdir()):
            if not subdir.is_dir():
                continue

            nome_classe = subdir.name

            if nome_classe not in MAPA_CLASSES:
                # Avisa sobre diretórios inesperados mas não interrompe
                print(f"[AVISO] Subdiretório '{nome_classe}' não reconhecido — ignorado.")
                continue

            rotulo = MAPA_CLASSES[nome_classe]

            for arquivo in subdir.iterdir():
                if arquivo.suffix.lower() in self.extensoes_validas:
                    self.caminhos_imagens.append(arquivo)
                    self.rotulos.append(rotulo)

        if len(self.caminhos_imagens) == 0:
            raise RuntimeError(f"Nenhuma imagem encontrada em {self.caminho_raiz}")

    def __len__(self) -> int:
        """Retorna o número total de amostras no dataset."""
        return len(self.caminhos_imagens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Carrega e retorna a imagem e o rótulo no índice idx.

        Parâmetros
        ----------
        idx : int
            Índice da amostra.

        Retorno
        -------
        tuple[torch.Tensor, int]
            Tensor da imagem (C, H, W) e rótulo inteiro da classe.
        """
        caminho = self.caminhos_imagens[idx]
        rotulo  = self.rotulos[idx]

        # Abre e converte para RGB (garante 3 canais mesmo para PNGs com alpha)
        imagem = Image.open(caminho).convert("RGB")

        if self.transformacoes:
            imagem = self.transformacoes(imagem)

        return imagem, rotulo

    def distribuicao_classes(self) -> dict[str, int]:
        """
        Retorna um dicionário com a contagem de imagens por classe.

        Retorno
        -------
        dict[str, int]
            Chave: nome legível da classe. Valor: contagem de imagens.
        """
        contagem = {nome: 0 for nome in NOMES_CLASSES}
        for rotulo in self.rotulos:
            contagem[NOMES_CLASSES[rotulo]] += 1
        return contagem


# ============================================================
# FUNÇÃO AUXILIAR: cria os DataLoaders prontos para uso
# ============================================================

def criar_dataloaders(
    tamanho_lote: int = TAMANHO_LOTE,
    num_workers: int = 2,
    semente: int = SEMENTE_ALEATORIA,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Instancia os três DataLoaders (treino, validação, teste) com suas
    respectivas transformações.

    Parâmetros
    ----------
    tamanho_lote : int
        Número de amostras por batch.
    num_workers : int
        Processos paralelos para carregamento de dados.
    semente : int
        Semente para o gerador aleatório do DataLoader.

    Retorno
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        (loader_treino, loader_validacao, loader_teste)
    """
    # Função auxiliar para fixar a semente em cada worker
    def _seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

    gerador = torch.Generator()
    gerador.manual_seed(semente)

    dataset_treino = DatasetPulmao(
        caminho_raiz=CAMINHO_TREINO,
        transformacoes=obter_transformacoes("treino"),
    )
    dataset_validacao = DatasetPulmao(
        caminho_raiz=CAMINHO_VALIDACAO,
        transformacoes=obter_transformacoes("validacao"),
    )
    dataset_teste = DatasetPulmao(
        caminho_raiz=CAMINHO_TESTE,
        transformacoes=obter_transformacoes("teste"),
    )

    loader_treino = DataLoader(
        dataset_treino,
        batch_size=tamanho_lote,
        shuffle=True,           # Embaralha a cada época
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=gerador,
        pin_memory=True,        # Acelera transferência CPU→GPU
    )
    loader_validacao = DataLoader(
        dataset_validacao,
        batch_size=tamanho_lote,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    loader_teste = DataLoader(
        dataset_teste,
        batch_size=tamanho_lote,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader_treino, loader_validacao, loader_teste
