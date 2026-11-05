"""
ReputationManager
=================
Implementa o mecanismo de defesa baseado em **desafios** e **reputação**
descrito na dissertação.

Funcionamento
-------------
Cada nó mantém um registro de reputação para cada par (peer) conhecido.
A reputação de um par j, vista pelo nó i, é calculada como:

    rep(j) = s_ij / (s_ij + f_ij + 1)   [suavização de Laplace]

onde:
  - s_ij  = número de desafios respondidos **corretamente** por j
  - f_ij  = número de desafios respondidos **incorretamente** por j

Fluxo
-----
1. Quando o nó recebe um **conselho** de um par, ele verifica se já tem
   ground-truth daquela amostra (desafio).
2. Se sim, compara a decisão do par com o gabarito e atualiza s/f.
3. Calcula a nova reputação.
4. Se rep < τ  E  (s+f) >= min_challenges  →  par é **ISOLATED**.
5. Pares ISOLATED são excluídos da lista de candidatos a conselheiros.

Parâmetros de configuração (via peer_config.json ou argumento direto)
----------------------------------------------------------------------
  tau              : limiar de reputação mínima (padrão 0.6)
  min_challenges   : nº mínimo de desafios para avaliar (padrão 10)

Integração
----------
O ReputationManager é criado no __init__ do CounselorNode e acessado:
  - pelo CounselorClient  → filtra peers isolados antes de sortear
  - pelo CounselorNode    → atualiza reputação após receber conselho

Logging
-------
Cada evento de desafio é registrado no logger via log_reputacao().
Métricas agregadas são salvas via log_metricas_defesa().
"""

import threading
import time
from datetime import datetime


# ---------------------------------------------------------------------------
# Estado de um par
# ---------------------------------------------------------------------------
ESTADO_EVALUATING = "EVALUATING"
ESTADO_TRUSTED    = "TRUSTED"
ESTADO_ISOLATED   = "ISOLATED"


class PeerReputation:
    """Mantém as contagens de sucessos/falhas de um único par."""

    def __init__(self, peer_name: str, peer_ip: str, peer_port: int):
        self.peer_name = peer_name
        self.peer_ip   = peer_ip
        self.peer_port = peer_port

        self.s_ij: int   = 0          # Acertos (sucessos)
        self.f_ij: int   = 0          # Erros   (falhas)
        self.estado: str = ESTADO_EVALUATING

    @property
    def n_challenges(self) -> int:
        return self.s_ij + self.f_ij

    def reputacao(self) -> float:
        """Reputação com suavização de Laplace: s / (s + f + 1)."""
        return self.s_ij / (self.s_ij + self.f_ij + 1)

    def __repr__(self):
        return (
            f"PeerReputation(name={self.peer_name}, s={self.s_ij}, f={self.f_ij}, "
            f"rep={self.reputacao():.4f}, estado={self.estado})"
        )


# ---------------------------------------------------------------------------
# Manager principal
# ---------------------------------------------------------------------------
class ReputationManager:
    """
    Gerencia a reputação de todos os pares conhecidos de um nó.

    Parâmetros
    ----------
    peers : list[dict]
        Lista de dicts com chaves 'name', 'ip', 'port' (vinda do ConfigManager).
    tau : float
        Limiar mínimo de reputação.  Par com rep < tau  →  ISOLATED.
    min_challenges : int
        Nº mínimo de desafios para habilitar o isolamento.
    logger : CounselorLogger ou None
        Instância do logger; se None, sem logging de CSV.
    node_id : str
        ID do nó local (para logging).
    """

    def __init__(
        self,
        peers: list,
        tau: float = 0.6,
        min_challenges: int = 10,
        logger=None,
        node_id: str = "unknown",
    ):
        self.tau            = tau
        self.min_challenges = min_challenges
        self.logger         = logger
        self.node_id        = node_id
        self._lock          = threading.Lock()

        # seq global de eventos (para o CSV de log)
        self._seq_event: int = 0

        # Dicionário: (ip, port) -> PeerReputation
        self._peers: dict[tuple, PeerReputation] = {}
        for p in peers:
            key = (p["ip"], p["port"])
            self._peers[key] = PeerReputation(
                peer_name=p.get("name", f"{p['ip']}:{p['port']}"),
                peer_ip=p["ip"],
                peer_port=p["port"],
            )

        print(
            f"[ReputationManager | {node_id}] Inicializado com {len(self._peers)} pares. "
            f"τ={tau}, min_challenges={min_challenges}"
        )

    # ------------------------------------------------------------------
    # Consulta de estado
    # ------------------------------------------------------------------
    def is_isolated(self, peer_ip: str, peer_port: int) -> bool:
        """Retorna True se o par está isolado."""
        with self._lock:
            pr = self._peers.get((peer_ip, peer_port))
            if pr is None:
                return False
            return pr.estado == ESTADO_ISOLATED

    def get_active_peers(self, all_peers: list) -> list:
        """
        Filtra a lista de peers, removendo os que estão ISOLATED.

        Parameters
        ----------
        all_peers : list[dict]
            Lista de dicts com 'ip' e 'port'.

        Returns
        -------
        list[dict]
            Pares não isolados.
        """
        with self._lock:
            filtered = []
            for p in all_peers:
                pr = self._peers.get((p["ip"], p["port"]))
                if pr is None or pr.estado != ESTADO_ISOLATED:
                    filtered.append(p)
            return filtered

    def get_all_states(self) -> dict:
        """Retorna snapshot do estado de todos os pares (para debug/log)."""
        with self._lock:
            return {
                f"{pr.peer_ip}:{pr.peer_port}": {
                    "name":     pr.peer_name,
                    "s":        pr.s_ij,
                    "f":        pr.f_ij,
                    "rep":      pr.reputacao(),
                    "estado":   pr.estado,
                    "n_chall":  pr.n_challenges,
                }
                for pr in self._peers.values()
            }

    # ------------------------------------------------------------------
    # Atualização de reputação
    # ------------------------------------------------------------------
    def register_challenge_result(
        self,
        peer_ip: str,
        peer_port: int,
        peer_decision: str,
        ground_truth: str,
        latencia_ms: float = 0.0,
    ) -> dict | None:
        """
        Registra o resultado de um desafio (comparação decisão × gabarito).

        A comparação é feita de forma normalizada (lowercase, strip).
        Um acerto ocorre quando peer_decision == ground_truth.

        Parâmetros
        ----------
        peer_ip, peer_port : identificação do par conselheiro.
        peer_decision      : o que o par respondeu (string de classe).
        ground_truth       : gabarito verdadeiro da amostra.
        latencia_ms        : tempo de resposta (para logging).

        Retorna
        -------
        dict com campos do evento (ou None se par não reconhecido).
        """
        with self._lock:
            pr = self._peers.get((peer_ip, peer_port))
            if pr is None:
                # Par desconhecido – não atualiza nada
                return None

            # Normaliza para comparação
            dec_norm = str(peer_decision).strip().lower()
            gt_norm  = str(ground_truth).strip().lower()

            # Acerto?
            acertou = (dec_norm == gt_norm)

            if acertou:
                pr.s_ij += 1
            else:
                pr.f_ij += 1

            rep   = pr.reputacao()
            prev  = pr.estado

            # Avaliação de isolamento
            if pr.n_challenges >= self.min_challenges and rep < self.tau:
                pr.estado = ESTADO_ISOLATED
            elif pr.n_challenges >= self.min_challenges and rep >= self.tau:
                pr.estado = ESTADO_TRUSTED
            # else: continua EVALUATING

            self._seq_event += 1

            evento = {
                "seq_event":      self._seq_event,
                "peer_name":      pr.peer_name,
                "peer_ip":        peer_ip,
                "peer_port":      peer_port,
                "s_ij":           pr.s_ij,
                "f_ij":           pr.f_ij,
                "reputacao":      rep,
                "limiar_tau":     self.tau,
                "min_desafios":   self.min_challenges,
                "estado":         pr.estado,
                "tipo_evento":    "CHALLENGE",
                "decisao_par":    peer_decision,
                "gabarito":       ground_truth,
                "latencia_ms":    latencia_ms,
            }

            if pr.estado == ESTADO_ISOLATED and prev != ESTADO_ISOLATED:
                print(
                    f"[ReputationManager | {self.node_id}] ⚠ PAR ISOLADO: {pr.peer_name} "
                    f"({peer_ip}:{peer_port})  rep={rep:.4f} < τ={self.tau} "
                    f"após {pr.n_challenges} desafios."
                )

            # Logging CSV
            if self.logger is not None:
                try:
                    self.logger.log_reputacao(evento)
                except Exception as e:
                    print(f"[ReputationManager] Erro ao logar reputação: {e}")

            return evento

    # ------------------------------------------------------------------
    # Relatório de métricas agregadas de defesa
    # ------------------------------------------------------------------
    def snapshot_metricas_defesa(
        self,
        total_normal_requests: int = 0,
    ) -> dict:
        """
        Calcula e retorna métricas agregadas do mecanismo de defesa.

        Returns
        -------
        dict com as métricas (também enviado ao logger se disponível).
        """
        with self._lock:
            total_challenges = sum(pr.n_challenges for pr in self._peers.values())
            total_isolated   = sum(1 for pr in self._peers.values() if pr.estado == ESTADO_ISOLATED)

            # As métricas de TP/FP/TN/FN só fazem sentido num contexto mais amplo
            # (onde sabemos quais pares são realmente maliciosos).
            # Aqui guardamos os valores brutos; o simulador pode preencher TP/FP etc.
            metricas = {
                "timestamp":            datetime.now().isoformat(timespec="seconds"),
                "total_normal_requests": total_normal_requests,
                "total_challenges":     total_challenges,
                "total_isolated_peers": total_isolated,
                # Placeholders – preenchidos externamente quando se conhece ground-truth dos pares
                "true_positives":  0,
                "false_positives": 0,
                "true_negatives":  0,
                "false_negatives": 0,
                "precision":       0.0,
                "recall":          0.0,
                "false_positive_rate": 0.0,
            }

            if self.logger is not None:
                try:
                    self.logger.log_metricas_defesa(metricas)
                except Exception as e:
                    print(f"[ReputationManager] Erro ao logar métricas: {e}")

            return metricas

