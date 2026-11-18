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

Estados de um par
-----------------
  EVALUATING → estado inicial; ainda não há desafios suficientes para julgar.
  TRUSTED    → rep >= τ após min_challenges desafios.
  ISOLATED   → rep < τ após min_challenges desafios.
               Par ISOLADO não é selecionado para conselhos normais,
               mas continua recebendo desafios de prova (probe challenges)
               a cada ``probe_every`` desafios válidos globais.

Mecanismo de Comeback (Reabilitação)
-------------------------------------
Um par ISOLATED não é descartado permanentemente. A rede continua lhe
enviando desafios probe para verificar se o comportamento melhorou
(ex.: ataque momentâneo cessou ou taxa de poison_rate foi reduzida).

  1. A cada ``probe_every`` desafios válidos (s+f de *qualquer* par),
     o ReputationManager registra internamente que um probe está pendente
     para cada par ISOLATED.
  2. Quando o par isolado responder um probe e acertar, incrementa
     ``comeback_hits``. Ao atingir ``comeback_threshold`` acertos
     consecutivos, o estado volta para EVALUATING e os contadores
     históricos s/f são preservados (a reputação deve ser reconquistada
     ao longo dos desafios seguintes com o histórico completo).
  3. Um erro durante o comeback zera o contador ``comeback_hits``
     (os consecutivos têm que ser limpos).
  4. O agendamento de probes é responsabilidade do CounselorClient:
     ao selecionar conselheiros, ele pode incluir um par ISOLATED
     se ``needs_probe()`` retornar True para ele.

Labels operacionais × Labels de classe
---------------------------------------
Decisões como "LOOP_CLOSED", "ERROR_*", "N/A" são labels internas da
rede de conselheiros — não são previsões de classe. O ReputationManager
IGNORA tais decisões via ``is_operational_label()``, não contabilizando
acertos nem falhas, e portanto não prejudicando a reputação de nós
honestos cujas respostas ficaram em loop ou falharam por razões de rede.

Parâmetros de configuração (via peer_config.json ou argumento direto)
----------------------------------------------------------------------
  tau              : limiar de reputação mínima (padrão 0.6)
  min_challenges   : nº mínimo de desafios para avaliar (padrão 10)
  probe_every      : a cada N desafios válidos globais, agenda um probe
                     para cada par ISOLATED (padrão 5)
  comeback_threshold: acertos consecutivos necessários para reabilitar
                     um par ISOLATED (padrão 3)

Integração
----------
O ReputationManager é criado no __init__ do CounselorNode e acessado:
  - pelo CounselorClient  → filtra peers isolados e seleciona probes
  - pelo CounselorNode    → atualiza reputação após receber conselho

Logging
-------
Cada evento de desafio é registrado no logger via log_reputacao().
Métricas agregadas são salvas via log_metricas_defesa().
"""

import threading
from datetime import datetime


# ---------------------------------------------------------------------------
# Labels operacionais — NÃO são previsões de classe e DEVEM ser ignoradas
# na atualização de reputação.
# ---------------------------------------------------------------------------
OPERATIONAL_LABELS = frozenset({
    "loop_closed",
    "error_connection",
    "error_timeout",
    "error_connection_refused",
    "error_classification",
    "error_response",
    "error",
    "n/a",
    "unknown",
    "none",
    "",
})


def is_operational_label(label: str) -> bool:
    """
    Retorna True se a string é uma label operacional/interna da rede
    (ex.: 'LOOP_CLOSED', 'ERROR_CONNECTION') e NÃO deve ser usada
    para avaliar a reputação de um par.
    """
    if label is None:
        return True
    normalized = str(label).strip().lower()
    if normalized in OPERATIONAL_LABELS:
        return True
    # Qualquer string que comece com "error" é operacional
    if normalized.startswith("error"):
        return True
    return False


# ---------------------------------------------------------------------------
# Estado de um par
# ---------------------------------------------------------------------------
ESTADO_EVALUATING = "EVALUATING"
ESTADO_TRUSTED    = "TRUSTED"
ESTADO_ISOLATED   = "ISOLATED"


class PeerReputation:
    """Mantém as contagens de sucessos/falhas e o estado de comeback de um par."""

    def __init__(self, peer_name: str, peer_ip: str, peer_port: int):
        self.peer_name = peer_name
        self.peer_ip   = peer_ip
        self.peer_port = peer_port

        self.s_ij: int   = 0          # Acertos históricos (sucesso)
        self.f_ij: int   = 0          # Erros históricos (falha)
        self.estado: str = ESTADO_EVALUATING

        # --- Comeback ---
        # Acertos consecutivos enquanto ISOLATED (probe challenges)
        self.comeback_hits: int    = 0
        # Número de probes pendentes (agendados pelo gestor global)
        self._probes_pending: int  = 0

    @property
    def n_challenges(self) -> int:
        return self.s_ij + self.f_ij

    def reputacao(self) -> float:
        """Reputação com suavização de Laplace: s / (s + f + 1)."""
        return self.s_ij / (self.s_ij + self.f_ij + 1)

    def needs_probe(self) -> bool:
        """True se há pelo menos um probe pendente para este par isolado."""
        return self.estado == ESTADO_ISOLATED and self._probes_pending > 0

    def consume_probe(self):
        """Marca um probe como consumido (chamado ao selecionar o par para probe)."""
        if self._probes_pending > 0:
            self._probes_pending -= 1

    def schedule_probe(self):
        """Agenda mais um probe para este par."""
        self._probes_pending += 1

    def __repr__(self):
        return (
            f"PeerReputation(name={self.peer_name}, s={self.s_ij}, f={self.f_ij}, "
            f"rep={self.reputacao():.4f}, estado={self.estado}, "
            f"comeback_hits={self.comeback_hits}, probes_pending={self._probes_pending})"
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
    probe_every : int
        A cada N desafios válidos globais, agenda um probe para cada par ISOLATED.
    comeback_threshold : int
        Acertos consecutivos de probe necessários para reabilitar um par ISOLATED.
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
        probe_every: int = 5,
        comeback_threshold: int = 3,
        logger=None,
        node_id: str = "unknown",
    ):
        self.tau                = tau
        self.min_challenges     = min_challenges
        self.probe_every        = probe_every
        self.comeback_threshold = comeback_threshold
        self.logger             = logger
        self.node_id            = node_id
        self._lock              = threading.Lock()

        # seq global de eventos (para o CSV de log)
        self._seq_event: int = 0

        # Contador global de desafios válidos (para agendar probes)
        self._global_valid_challenges: int = 0

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
            f"τ={tau}, min_challenges={min_challenges}, "
            f"probe_every={probe_every}, comeback_threshold={comeback_threshold}"
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
        Peers ISOLATED com probe pendente são INCLUÍDOS de volta na lista —
        o caller (CounselorClient) deve marcar o probe como consumido ao usá-los.

        Parameters
        ----------
        all_peers : list[dict]
            Lista de dicts com 'ip' e 'port'.

        Returns
        -------
        list[dict]
            Pares ativos (não isolados) OU isolados com probe pendente.
        """
        with self._lock:
            filtered = []
            for p in all_peers:
                pr = self._peers.get((p["ip"], p["port"]))
                if pr is None:
                    filtered.append(p)
                elif pr.estado != ESTADO_ISOLATED:
                    filtered.append(p)
                elif pr.needs_probe():
                    # Par isolado, mas com probe agendado: inclui para o desafio
                    filtered.append(p)
            return filtered

    def get_peers_needing_probe(self) -> list:
        """
        Retorna lista de peers ISOLATED que possuem probe pendente.
        Útil para o cliente priorizar esses pares como candidatos a probe.
        """
        with self._lock:
            return [
                {"ip": pr.peer_ip, "port": pr.peer_port, "name": pr.peer_name}
                for pr in self._peers.values()
                if pr.needs_probe()
            ]

    def consume_probe_for(self, peer_ip: str, peer_port: int):
        """Marca um probe pendente como consumido para o par especificado."""
        with self._lock:
            pr = self._peers.get((peer_ip, peer_port))
            if pr is not None:
                pr.consume_probe()

    def get_all_states(self) -> dict:
        """Retorna snapshot do estado de todos os pares (para debug/log)."""
        with self._lock:
            return {
                f"{pr.peer_ip}:{pr.peer_port}": {
                    "name":           pr.peer_name,
                    "s":              pr.s_ij,
                    "f":              pr.f_ij,
                    "rep":            pr.reputacao(),
                    "estado":         pr.estado,
                    "n_chall":        pr.n_challenges,
                    "comeback_hits":  pr.comeback_hits,
                    "probes_pending": pr._probes_pending,
                }
                for pr in self._peers.values()
            }

    # ------------------------------------------------------------------
    # Agendamento interno de probes (chamado a cada desafio válido)
    # ------------------------------------------------------------------
    def _maybe_schedule_probes(self):
        """
        Se o contador global de desafios válidos atingiu um múltiplo de
        ``probe_every``, agenda um probe para cada par ISOLATED.
        Deve ser chamado COM o lock já adquirido.
        """
        if self.probe_every <= 0:
            return
        if self._global_valid_challenges % self.probe_every == 0:
            for pr in self._peers.values():
                if pr.estado == ESTADO_ISOLATED:
                    pr.schedule_probe()
                    print(
                        f"[ReputationManager | {self.node_id}] 🔍 Probe agendado para "
                        f"{pr.peer_name} ({pr.peer_ip}:{pr.peer_port}) — "
                        f"comeback_hits={pr.comeback_hits}/{self.comeback_threshold}"
                    )

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
        tipo_evento_override: str | None = None,
    ) -> dict | None:
        """
        Registra o resultado de um desafio (comparação decisão × gabarito).

        IMPORTANTE: Decisões operacionais (LOOP_CLOSED, ERROR_*, N/A, etc.)
        são silenciosamente ignoradas — não contabilizam acerto nem falha —
        garantindo que nós honestos não sejam penalizados por falhas de rede.

        A comparação é feita de forma normalizada (lowercase, strip).
        Um acerto ocorre quando peer_decision == ground_truth.

        Parâmetros
        ----------
        peer_ip, peer_port      : identificação do par conselheiro.
        peer_decision           : o que o par respondeu (string de classe).
        ground_truth            : gabarito verdadeiro da amostra.
        latencia_ms             : tempo de resposta (para logging).
        tipo_evento_override    : se informado, substitui o tipo_evento inferido
                                  internamente (ex.: "ACTIVE_CHALLENGE" para
                                  desafios proativos disparados por contagem de
                                  amostras).

        Retorna
        -------
        dict com campos do evento (ou None se par não reconhecido ou decisão ignorada).
        """
        # --- Filtra labels operacionais ANTES de adquirir o lock ---
        if is_operational_label(peer_decision):
            # Decisão interna da rede; não afeta reputação de ninguém.
            return None

        gt_norm = str(ground_truth).strip().lower()
        if gt_norm in ("n/a", "n/a_from_peer", "unknown", ""):
            # Ground truth inválido; não há como avaliar.
            return None

        with self._lock:
            pr = self._peers.get((peer_ip, peer_port))
            if pr is None:
                # Par desconhecido – não atualiza nada
                return None

            dec_norm = str(peer_decision).strip().lower()
            acertou  = (dec_norm == gt_norm)

            prev_estado = pr.estado

            if pr.estado == ESTADO_ISOLATED:
                # ---- Modo Probe / Comeback ----
                # Desafio chega ao par isolado; avalia acertos consecutivos.
                if acertou:
                    pr.comeback_hits += 1
                    pr.s_ij += 1
                    tipo_evento = tipo_evento_override or "PROBE_HIT"
                    if pr.comeback_hits >= self.comeback_threshold:
                        # Reabilitação: volta para EVALUATING com histórico
                        pr.estado       = ESTADO_EVALUATING
                        pr.comeback_hits = 0
                        print(
                            f"[ReputationManager | {self.node_id}] ✅ COMEBACK: {pr.peer_name} "
                            f"({peer_ip}:{peer_port}) reabilitado → EVALUATING. "
                            f"rep={pr.reputacao():.4f} (histórico: s={pr.s_ij} f={pr.f_ij})"
                        )
                else:
                    # Erro durante probe: zera os consecutivos
                    pr.comeback_hits = 0
                    pr.f_ij += 1
                    tipo_evento = tipo_evento_override or "PROBE_MISS"

            else:
                # ---- Modo normal (EVALUATING ou TRUSTED) ----
                if acertou:
                    pr.s_ij += 1
                else:
                    pr.f_ij += 1
                tipo_evento = tipo_evento_override or "CHALLENGE"

            rep = pr.reputacao()

            # Avaliação de isolamento (apenas fora do modo probe, pois
            # durante probe o estado já é ISOLATED e é o comeback que decide)
            if pr.estado != ESTADO_ISOLATED:
                if pr.n_challenges >= self.min_challenges and rep < self.tau:
                    pr.estado = ESTADO_ISOLATED
                    pr.comeback_hits = 0   # garante reset ao entrar no isolamento
                    print(
                        f"[ReputationManager | {self.node_id}] ⚠ PAR ISOLADO: {pr.peer_name} "
                        f"({peer_ip}:{peer_port})  rep={rep:.4f} < τ={self.tau} "
                        f"após {pr.n_challenges} desafios."
                    )
                elif pr.n_challenges >= self.min_challenges and rep >= self.tau:
                    pr.estado = ESTADO_TRUSTED

            # Incrementa contador global e agenda probes se necessário
            self._global_valid_challenges += 1
            self._maybe_schedule_probes()

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
                "tipo_evento":    tipo_evento,
                "decisao_par":    peer_decision,
                "gabarito":       ground_truth,
                "latencia_ms":    latencia_ms,
                "comeback_hits":  pr.comeback_hits,
            }

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
            total_trusted    = sum(1 for pr in self._peers.values() if pr.estado == ESTADO_TRUSTED)

            # As métricas de TP/FP/TN/FN só fazem sentido num contexto mais amplo
            # (onde sabemos quais pares são realmente maliciosos).
            # Aqui guardamos os valores brutos; o simulador pode preencher TP/FP etc.
            metricas = {
                "timestamp":            datetime.now().isoformat(timespec="seconds"),
                "total_normal_requests": total_normal_requests,
                "total_challenges":     total_challenges,
                "total_isolated_peers": total_isolated,
                "total_trusted_peers":  total_trusted,
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
