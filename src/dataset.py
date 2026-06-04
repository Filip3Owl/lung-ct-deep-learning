"""
dataset.py
----------
Dataset customizado PyTorch para imagens de tomografia computadorizada pulmonar.

Responsabilidades
-----------------
- Carregar imagens a partir de subdiretórios organizados por classe
- Aplicar transformações distintas para treino (augmentações) e inferência
- Fornecer DataLoaders prontos para uso nos notebooks de treinamento e avaliação

Estrutura de diretórios esperada
---------------------------------
    split/
        <nome_classe_a>/
            imagem1.png
            imagem2.png
        <nome_classe_b>/
            ...

Os nomes das subpastas devem corresponder às chaves de MAPA_CLASSES em config.py.

Exportações principais
----------------------
obter_transformacoes(modo)          Pipeline de transforms por modo de uso
DatasetPulmao                       Dataset PyTorch customizado
criar_dataloaders(...)              Instancia os três DataLoaders de uma vez

Uso
---
    from dataset import criar_dataloaders

    loader_treino, loader_val, loader_teste = criar_dataloaders(tamanho_lote=32)
    imagens, rotulos = next(iter(loader_treino))  # (32, 3, 224, 224), (32,)
"""

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
    MAPA_CLASSES, NOMES_CLASSES,
    TAMANHO_LOTE, SEMENTE_ALEATORIA,
)


# ============================================================
# TRANSFORMAÇÕES (pipeline de pré-processamento / augmentação)
# ============================================================

def obter_transformacoes(modo: str) -> T.Compose:
    """
    Retorna o pipeline de transformações adequado para cada fase de uso.

    Modos disponíveis
    -----------------
    'treino'
        Inclui augmentações aleatórias (flip, rotação, jitter de cor, translação)
        para regularizar o modelo e aumentar artificialmente a diversidade do
        dataset pequeno. Aumentações agressivas (distorções elásticas, crop extremo)
        são evitadas para não destruir características radiológicas relevantes.
    'validacao' / 'teste'
        Apenas redimensiona e normaliza — sem aleatoriedade para garantir
        reprodutibilidade das métricas de avaliação.

    Parâmetros
    ----------
    modo : str
        Um de 'treino', 'validacao' ou 'teste'.

    Retorno
    -------
    torchvision.transforms.Compose
        Pipeline de transformações encadeadas.

    Raises
    ------
    ValueError
        Se `modo` não for um dos valores reconhecidos.
    """
    if modo not in {"treino", "validacao", "teste"}:
        raise ValueError(f"Modo inválido: '{modo}'. Use 'treino', 'validacao' ou 'teste'.")

    if modo == "treino":
        return T.Compose([
            T.Resize((ALTURA_IMAGEM, LARGURA_IMAGEM)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            T.ToTensor(),
            T.Normalize(mean=MEDIA_IMAGENET, std=DESVIO_IMAGENET),
        ])

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
    Dataset PyTorch para classificação de imagens de TC pulmonar.

    Percorre os subdiretórios de `caminho_raiz`, associa cada imagem
    ao índice de classe correspondente via MAPA_CLASSES e aplica as
    transformações fornecidas em cada acesso.

    Parâmetros
    ----------
    caminho_raiz : Path | str
        Diretório raiz do split (treino, validação ou teste).
    transformacoes : callable, optional
        Pipeline de transformações (ex.: retorno de obter_transformacoes()).
    extensoes_validas : tuple[str], optional
        Extensões de arquivo reconhecidas como imagens. Padrão: .png, .jpg, .jpeg.

    Atributos públicos
    ------------------
    caminhos_imagens : list[Path]
        Caminhos absolutos de todas as imagens carregadas.
    rotulos : list[int]
        Índice de classe correspondente a cada imagem (mesma ordem).

    Raises
    ------
    FileNotFoundError
        Se `caminho_raiz` não existir no sistema de arquivos.
    RuntimeError
        Se nenhuma imagem válida for encontrada em `caminho_raiz`.
    """

    EXTENSOES_VALIDAS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")

    def __init__(
        self,
        caminho_raiz: Path | str,
        transformacoes: Optional[Callable] = None,
        extensoes_validas: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.caminho_raiz      = Path(caminho_raiz)
        self.transformacoes    = transformacoes
        self.extensoes_validas = extensoes_validas or self.EXTENSOES_VALIDAS

        self.caminhos_imagens: list[Path] = []
        self.rotulos:          list[int]  = []

        self._carregar_dataset()

    # ----------------------------------------------------------
    # Carregamento interno
    # ----------------------------------------------------------

    def _carregar_dataset(self) -> None:
        """
        Percorre subdiretórios de caminho_raiz e popula caminhos_imagens / rotulos.
        Subdiretórios cujo nome não conste em MAPA_CLASSES são ignorados com aviso.
        """
        if not self.caminho_raiz.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {self.caminho_raiz}")

        for subdir in sorted(self.caminho_raiz.iterdir()):
            if not subdir.is_dir():
                continue

            if subdir.name not in MAPA_CLASSES:
                print(f"[AVISO] Subdiretório '{subdir.name}' não reconhecido em MAPA_CLASSES — ignorado.")
                continue

            rotulo = MAPA_CLASSES[subdir.name]

            for arquivo in subdir.iterdir():
                if arquivo.suffix.lower() in self.extensoes_validas:
                    self.caminhos_imagens.append(arquivo)
                    self.rotulos.append(rotulo)

        if not self.caminhos_imagens:
            raise RuntimeError(f"Nenhuma imagem encontrada em: {self.caminho_raiz}")

    # ----------------------------------------------------------
    # Interface Dataset
    # ----------------------------------------------------------

    def __len__(self) -> int:
        """Número total de amostras no dataset."""
        return len(self.caminhos_imagens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Carrega e retorna a imagem e o rótulo no índice `idx`.

        A imagem é convertida para RGB antes das transformações, garantindo
        3 canais mesmo para arquivos PNG com canal alpha.

        Parâmetros
        ----------
        idx : int
            Índice da amostra.

        Retorno
        -------
        tuple[torch.Tensor, int]
            Tensor da imagem com shape (3, H, W) e índice inteiro da classe.
        """
        imagem = Image.open(self.caminhos_imagens[idx]).convert("RGB")
        rotulo = self.rotulos[idx]

        if self.transformacoes:
            imagem = self.transformacoes(imagem)

        return imagem, rotulo

    # ----------------------------------------------------------
    # Utilitários
    # ----------------------------------------------------------

    def distribuicao_classes(self) -> dict[str, int]:
        """
        Contagem de imagens por classe.

        Retorno
        -------
        dict[str, int]
            {nome_legível_da_classe: contagem}
        """
        contagem = {nome: 0 for nome in NOMES_CLASSES}
        for rotulo in self.rotulos:
            contagem[NOMES_CLASSES[rotulo]] += 1
        return contagem


# ============================================================
# FUNÇÃO DE CONVENIÊNCIA
# ============================================================

def criar_dataloaders(
    tamanho_lote: int = TAMANHO_LOTE,
    num_workers:  int = 2,
    semente:      int = SEMENTE_ALEATORIA,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Instancia e retorna os DataLoaders de treino, validação e teste.

    Aplica automaticamente as transformações corretas para cada split:
    augmentações no treino, apenas normalização na validação e no teste.
    A semente é fixada no gerador do DataLoader e nos workers para garantir
    reprodutibilidade do embaralhamento entre execuções.

    Parâmetros
    ----------
    tamanho_lote : int
        Número de amostras por batch.
    num_workers : int
        Processos paralelos para carregamento de dados. Use 0 para debug
        (execução no processo principal, sem multiprocessing).
    semente : int
        Semente aleatória para reprodutibilidade.

    Retorno
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        (loader_treino, loader_validacao, loader_teste)

    Exemplo
    -------
        loader_treino, loader_val, loader_teste = criar_dataloaders()
        for imagens, rotulos in loader_treino:
            ...  # imagens: (B, 3, 224, 224), rotulos: (B,)
    """
    def _seed_worker(worker_id: int) -> None:
        """Propaga semente para numpy dentro de cada worker do DataLoader."""
        np.random.seed(torch.initial_seed() % 2**32)

    gerador = torch.Generator()
    gerador.manual_seed(semente)

    loader_treino = DataLoader(
        DatasetPulmao(CAMINHO_TREINO, obter_transformacoes("treino")),
        batch_size=tamanho_lote,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=gerador,
        pin_memory=True,
    )
    loader_validacao = DataLoader(
        DatasetPulmao(CAMINHO_VALIDACAO, obter_transformacoes("validacao")),
        batch_size=tamanho_lote,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    loader_teste = DataLoader(
        DatasetPulmao(CAMINHO_TESTE, obter_transformacoes("teste")),
        batch_size=tamanho_lote,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader_treino, loader_validacao, loader_teste
