"""
treinamento.py
--------------
Loop de treinamento e validação do modelo de classificação pulmonar.

Funcionalidades
---------------
- EarlyStopping: monitora a loss de validação e interrompe quando não há
  melhora, evitando overfitting com o dataset pequeno.
- Checkpointing automático: salva o melhor modelo a cada época de melhora.
- Logging tabular: exibe métricas por época de forma legível no notebook.
- Suporte a ReduceLROnPlateau e schedulers convencionais sem configuração extra.

Exportações principais
----------------------
EarlyStopping          Classe de monitoramento e checkpointing
treinar_modelo(...)    Loop completo de treino + validação

Uso
---
    from treinamento import treinar_modelo
    import torch.optim as optim
    import torch.nn as nn

    criterio  = nn.CrossEntropyLoss(weight=pesos_classes)
    otimizador = optim.Adam(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(otimizador, patience=3)

    historico = treinar_modelo(
        modelo, loader_treino, loader_val,
        criterio, otimizador, scheduler, epocas=10,
    )
"""

import time
import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CAMINHO_MODELOS, NOME_ARQUIVO_MODELO, EPOCAS, PACIENCIA


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """
    Monitora a loss de validação e sinaliza parada quando não há melhora.

    A cada época, compara a loss atual com a melhor registrada. Se a melhora
    for menor que `delta`, incrementa um contador. Quando o contador atinge
    `paciencia`, define `self.parar = True`. O melhor estado do modelo é
    salvo em disco e pode ser restaurado ao final do treino.

    Parâmetros
    ----------
    paciencia : int
        Épocas consecutivas sem melhora antes de parar.
    delta : float
        Redução mínima na loss para ser considerada melhora real.
    caminho_checkpoint : Path
        Arquivo onde o melhor state_dict será persistido.

    Atributos públicos
    ------------------
    parar : bool
        True quando o critério de parada foi atingido.
    melhor_loss : float
        Menor loss de validação observada até o momento.
    contador : int
        Épocas consecutivas sem melhora desde a última melhor época.
    """

    def __init__(
        self,
        paciencia:          int  = PACIENCIA,
        delta:              float = 1e-4,
        caminho_checkpoint: Path  = CAMINHO_MODELOS / NOME_ARQUIVO_MODELO,
    ) -> None:
        self.paciencia          = paciencia
        self.delta              = delta
        self.caminho_checkpoint = caminho_checkpoint

        self.contador:     int            = 0
        self.melhor_loss:  float          = float("inf")
        self.parar:        bool           = False
        self._melhor_pesos: Optional[dict] = None

    def __call__(self, loss_val: float, modelo: nn.Module) -> None:
        """
        Avalia a loss da época atual e atualiza o estado interno.

        Salva o modelo em disco quando a loss melhora. Incrementa o
        contador (e eventualmente ativa `self.parar`) quando não melhora.

        Parâmetros
        ----------
        loss_val : float
            Loss de validação da época atual.
        modelo : nn.Module
            Modelo a ser salvo caso seja o melhor até agora.
        """
        if loss_val < self.melhor_loss - self.delta:
            self.melhor_loss   = loss_val
            self._melhor_pesos = copy.deepcopy(modelo.state_dict())
            self.contador      = 0
            torch.save(self._melhor_pesos, self.caminho_checkpoint)
        else:
            self.contador += 1
            if self.contador >= self.paciencia:
                self.parar = True

    def restaurar_melhor_modelo(self, modelo: nn.Module) -> nn.Module:
        """
        Carrega o state_dict do melhor checkpoint de volta no modelo.

        Deve ser chamado ao final do loop de treino para garantir que o
        modelo retornado corresponde à melhor época, não à última.

        Parâmetros
        ----------
        modelo : nn.Module
            Instância onde os pesos serão restaurados in-place.

        Retorno
        -------
        nn.Module
            O mesmo objeto `modelo`, com os melhores pesos carregados.
        """
        if self._melhor_pesos is not None:
            modelo.load_state_dict(self._melhor_pesos)
        return modelo


# ============================================================
# FUNÇÃO INTERNA DE ÉPOCA
# ============================================================

def _rodar_epoca(
    modelo:      nn.Module,
    loader:      DataLoader,
    criterio:    nn.Module,
    otimizador:  Optional[Optimizer],
    dispositivo: torch.device,
    modo_treino: bool,
) -> Tuple[float, float]:
    """
    Executa uma única passagem pelo DataLoader (treino ou avaliação).

    No modo treino, o gradiente é calculado e os pesos são atualizados.
    No modo avaliação, executa sob torch.no_grad() para economia de memória.

    Parâmetros
    ----------
    modelo : nn.Module
    loader : DataLoader
    criterio : nn.Module
        Função de perda (ex.: CrossEntropyLoss).
    otimizador : Optimizer | None
        None no modo avaliação.
    dispositivo : torch.device
    modo_treino : bool
        True ativa model.train() e backprop; False ativa model.eval().

    Retorno
    -------
    tuple[float, float]
        (loss_média_da_época, acurácia_da_época)
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
            acertos        += (saidas.argmax(dim=1) == rotulos).sum().item()
            total          += imagens.size(0)

    return loss_acumulada / total, acertos / total


# ============================================================
# LOOP DE TREINAMENTO PRINCIPAL
# ============================================================

def treinar_modelo(
    modelo:           nn.Module,
    loader_treino:    DataLoader,
    loader_validacao: DataLoader,
    criterio:         nn.Module,
    otimizador:       Optimizer,
    scheduler:        Optional[LRScheduler] = None,
    epocas:           int                   = EPOCAS,
    paciencia:        int                   = PACIENCIA,
    dispositivo:      Optional[torch.device] = None,
) -> Dict[str, list]:
    """
    Loop completo de treinamento com validação, Early Stopping e checkpointing.

    Fluxo por época
    ---------------
    1. Fase de treino  — forward + backward + atualização dos pesos
    2. Fase de validação — forward apenas (sem atualizar pesos)
    3. Scheduler step — ReduceLROnPlateau recebe a loss de validação;
       outros schedulers avançam sem métrica
    4. EarlyStopping — salva o modelo se houve melhora ou incrementa contador

    Parâmetros
    ----------
    modelo : nn.Module
        Modelo a treinar (já movido para o dispositivo).
    loader_treino : DataLoader
    loader_validacao : DataLoader
    criterio : nn.Module
        Função de perda (ex.: CrossEntropyLoss com pesos de classe).
    otimizador : Optimizer
        Otimizador configurado com os parâmetros do modelo.
    scheduler : LRScheduler, optional
        Agendador de taxa de aprendizado. Detecta automaticamente
        ReduceLROnPlateau e passa a loss de validação quando necessário.
    epocas : int
        Número máximo de épocas antes de encerrar.
    paciencia : int
        Paciência do Early Stopping (épocas sem melhora).
    dispositivo : torch.device, optional
        Detectado automaticamente (CUDA se disponível, senão CPU).

    Retorno
    -------
    dict[str, list]
        Histórico de métricas com as chaves:
        'loss_treino', 'loss_validacao',
        'acuracia_treino', 'acuracia_validacao', 'lr'

    Notas
    -----
    O modelo é restaurado para os pesos da melhor época antes do retorno,
    independentemente do critério de parada (Early Stopping ou esgotamento
    de épocas).
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

    print(f"\nDispositivo: {dispositivo}")
    print(f"{'Época':>6} | {'LR':>10} | {'L.Treino':>10} | {'Ac.Treino':>10} | {'L.Valid':>9} | {'Ac.Valid':>9}")
    print("-" * 72)

    for epoca in range(1, epocas + 1):
        t0 = time.time()

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

        print(
            f"{epoca:>6}/{epocas} | {lr_atual:>10.2e} | {loss_t:>10.4f} | "
            f"{ac_t:>10.4f} | {loss_v:>9.4f} | {ac_v:>9.4f}  [{time.time()-t0:.1f}s]"
        )

        if scheduler is not None:
            # ReduceLROnPlateau exige a métrica de referência; demais schedulers não
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss_v)
            else:
                scheduler.step()

        early_stopping(loss_v, modelo)

        if early_stopping.parar:
            print(f"\nEarly Stopping na época {epoca}. Melhor loss val: {early_stopping.melhor_loss:.4f}")
            break

    modelo = early_stopping.restaurar_melhor_modelo(modelo)
    return historico
