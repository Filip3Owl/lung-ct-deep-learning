"""
treinamento.py
--------------
Contém o loop de treinamento e validação do modelo.
Implementa Early Stopping, salvamento do melhor modelo e logging de métricas.
"""

import time
import copy
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CAMINHO_MODELOS, NOME_ARQUIVO_MODELO, EPOCAS, PACIENCIA


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """
    Monitora a loss de validação e interrompe o treino quando não há melhora
    por 'paciencia' épocas consecutivas.

    Parâmetros
    ----------
    paciencia : int
        Número de épocas sem melhora antes de parar.
    delta : float
        Melhora mínima para ser considerada significativa.
    caminho_checkpoint : Path
        Onde salvar os pesos do melhor modelo.
    """

    def __init__(
        self,
        paciencia: int = PACIENCIA,
        delta: float = 1e-4,
        caminho_checkpoint: Path = CAMINHO_MODELOS / NOME_ARQUIVO_MODELO,
    ) -> None:
        self.paciencia = paciencia
        self.delta = delta
        self.caminho_checkpoint = caminho_checkpoint

        self.contador     = 0
        self.melhor_loss  = float("inf")
        self.parar        = False
        self.melhor_pesos: Optional[dict] = None

    def __call__(self, loss_val: float, modelo: nn.Module) -> None:
        """
        Avalia se houve melhora. Salva o modelo se sim, incrementa
        o contador caso contrário.

        Parâmetros
        ----------
        loss_val : float
            Loss de validação da época atual.
        modelo : nn.Module
            Modelo a ser salvo caso seja o melhor até agora.
        """
        if loss_val < self.melhor_loss - self.delta:
            self.melhor_loss  = loss_val
            self.melhor_pesos = copy.deepcopy(modelo.state_dict())
            self.contador     = 0
            torch.save(self.melhor_pesos, self.caminho_checkpoint)
        else:
            self.contador += 1
            if self.contador >= self.paciencia:
                self.parar = True

    def restaurar_melhor_modelo(self, modelo: nn.Module) -> nn.Module:
        """
        Carrega os pesos do melhor checkpoint salvo de volta no modelo.

        Parâmetros
        ----------
        modelo : nn.Module
            Instância do modelo onde os pesos serão restaurados.

        Retorno
        -------
        nn.Module
            Modelo com os melhores pesos restaurados.
        """
        if self.melhor_pesos is not None:
            modelo.load_state_dict(self.melhor_pesos)
        return modelo


# ============================================================
# FUNÇÕES DE ÉPOCA
# ============================================================

def _rodar_epoca(
    modelo: nn.Module,
    loader: DataLoader,
    criterio: nn.Module,
    otimizador: Optional[Optimizer],
    dispositivo: torch.device,
    modo_treino: bool,
) -> Tuple[float, float]:
    """
    Executa uma única época de treino ou avaliação.

    Parâmetros
    ----------
    modelo : nn.Module
        Modelo PyTorch.
    loader : DataLoader
        DataLoader da época.
    criterio : nn.Module
        Função de perda.
    otimizador : Optimizer | None
        Otimizador (None no modo avaliação).
    dispositivo : torch.device
        Dispositivo de execução.
    modo_treino : bool
        True para treino (atualiza pesos), False para avaliação.

    Retorno
    -------
    tuple[float, float]
        (loss_media, acuracia) da época.
    """
    modelo.train() if modo_treino else modelo.eval()

    loss_acumulada = 0.0
    acertos        = 0
    total          = 0

    ctx = torch.enable_grad() if modo_treino else torch.no_grad()

    with ctx:
        for imagens, rotulos in tqdm(loader, leave=False, desc="  Batch"):
            imagens = imagens.to(dispositivo, non_blocking=True)
            rotulos = rotulos.to(dispositivo, non_blocking=True)

            saidas = modelo(imagens)
            loss   = criterio(saidas, rotulos)

            if modo_treino:
                otimizador.zero_grad()
                loss.backward()
                otimizador.step()

            loss_acumulada += loss.item() * imagens.size(0)
            previsoes       = saidas.argmax(dim=1)
            acertos        += (previsoes == rotulos).sum().item()
            total          += imagens.size(0)

    loss_media = loss_acumulada / total
    acuracia   = acertos / total

    return loss_media, acuracia


# ============================================================
# LOOP DE TREINAMENTO PRINCIPAL
# ============================================================

def treinar_modelo(
    modelo: nn.Module,
    loader_treino: DataLoader,
    loader_validacao: DataLoader,
    criterio: nn.Module,
    otimizador: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    epocas: int = EPOCAS,
    paciencia: int = PACIENCIA,
    dispositivo: Optional[torch.device] = None,
) -> Dict:
    """
    Loop completo de treinamento com validação, Early Stopping e checkpointing.

    Parâmetros
    ----------
    modelo : nn.Module
        Modelo a treinar.
    loader_treino : DataLoader
        DataLoader do conjunto de treino.
    loader_validacao : DataLoader
        DataLoader do conjunto de validação.
    criterio : nn.Module
        Função de perda (ex.: CrossEntropyLoss).
    otimizador : Optimizer
        Otimizador (ex.: Adam, SGD).
    scheduler : _LRScheduler, optional
        Agendador de taxa de aprendizado.
    epocas : int
        Número máximo de épocas.
    paciencia : int
        Épocas de paciência para Early Stopping.
    dispositivo : torch.device, optional
        CPU ou CUDA. Detectado automaticamente se None.

    Retorno
    -------
    dict
        Histórico de métricas:
        {
            'loss_treino': [...],
            'loss_validacao': [...],
            'acuracia_treino': [...],
            'acuracia_validacao': [...],
            'lr': [...],
        }
    """
    if dispositivo is None:
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    early_stopping = EarlyStopping(paciencia=paciencia)

    historico: Dict[str, list] = {
        "loss_treino":        [],
        "loss_validacao":     [],
        "acuracia_treino":    [],
        "acuracia_validacao": [],
        "lr":                 [],
    }

    print(f"\nIniciando treinamento no dispositivo: {dispositivo}")
    print(f"{'Época':>6} | {'LR':>10} | {'L.Treino':>10} | {'Ac.Treino':>10} | {'L.Valid':>9} | {'Ac.Valid':>9}")
    print("-" * 72)

    for epoca in range(1, epocas + 1):
        inicio = time.time()

        loss_t, ac_t = _rodar_epoca(
            modelo, loader_treino, criterio, otimizador, dispositivo, modo_treino=True
        )
        loss_v, ac_v = _rodar_epoca(
            modelo, loader_validacao, criterio, None, dispositivo, modo_treino=False
        )

        lr_atual = otimizador.param_groups[0]["lr"]

        historico["loss_treino"].append(loss_t)
        historico["loss_validacao"].append(loss_v)
        historico["acuracia_treino"].append(ac_t)
        historico["acuracia_validacao"].append(ac_v)
        historico["lr"].append(lr_atual)

        duracao = time.time() - inicio

        print(
            f"{epoca:>6}/{epocas} | {lr_atual:>10.2e} | {loss_t:>10.4f} | "
            f"{ac_t:>10.4f} | {loss_v:>9.4f} | {ac_v:>9.4f}  [{duracao:.1f}s]"
        )

        # Ajusta o scheduler (ReduceLROnPlateau usa a métrica de validação)
        if scheduler is not None:
            if hasattr(scheduler, "step") and "metrics" in scheduler.step.__code__.co_varnames:
                scheduler.step(loss_v)
            else:
                scheduler.step()

        early_stopping(loss_v, modelo)

        if early_stopping.parar:
            print(f"\nEarly Stopping na época {epoca}. Melhor loss de val: {early_stopping.melhor_loss:.4f}")
            break

    # Restaura o melhor modelo antes de retornar
    modelo = early_stopping.restaurar_melhor_modelo(modelo)

    return historico
