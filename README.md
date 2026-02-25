# 🌐 CounselorNode 🌐

#### EN 🇺🇸 

### ✔️ Overview

CounselorNode is a fully decentralized peer-to-peer (P2P) implementation of a node from a Counselors Network (CN) for collaborative Intrusion Detection Systems (IDS).

Each node performs local classification using **Dynamic Classifier Selection (DCS)** and requests labeled advice from peers only when local decisions are unreliable due to conflict.

The tool supports recursive advice exchange, cycle-closure detection, and event instrumentation for reproducible experimentation.

---

## 📚 Index

- [Architecture](#architecture)
- [Test Environment](#test-environment)
- [Requirements](#requirements)
- [Installation](#installation)
- [Execution](#execution)
- [Configuration](#configuration)
- [🇧🇷 PT](#pt)

---

<a id="architecture"></a>
## 🏗️ Architecture

The project is organized into four main modules:

```
counselornode/
│
├── config/              # peer_config.json (node parameters)
├── core/                # Node logic and classifier engine
├── infrastructure/      # Networking, config manager, logger
└── run_node.py          # Command-line entry point
```

### Core Components

**ClassifierEngine**
- K-Means clustering
- Dynamic Classifier Selection (Decision Tree, KNN, SVM and Naive Bayes by default)
- Conflict detection
- Outlier detection

**CounselorNode**
- Executes local classification
- Triggers recursive advice requests
- Maintains a forwarding-node list to prevent cyclic requests
- Detects loop closure

**Networking Layer**
- TCP socket-based communication
- Advice request/response exchange
- Recursive forwarding control

---

<a id="test-environment"></a>
## 🖥️ Test Environment

The tool was tested under the following configurations:

| Setting | Environment I | Environment II |
|----------|----------------|----------------|
| OS | Windows 11 | Ubuntu 20.04 LTS |
| Processor | AMD Ryzen 7 5700X3D | Intel Core i5-10300H |
| RAM | 16 GB | 16 GB |
| Architecture | 64-bit | 64-bit |

---

<a id="requirements"></a>
## 📝 Requirements

CounselorNode is implemented in Python.

| Dependency | Recommended Version |
|------------|--------------------|
| Python | 3.9+ |
| Docker (optional) | 28.1.1+ |

Libraries are listed in:

```
requirements.txt
```

---

<a id="installation"></a>
## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/sequincozes/CounselorNode.git
cd CounselorNode
```

Create and activate a virtual environment (optional but recommended):

### Linux / macOS

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---
<a id="execution"></a>
## ▶️ Execution

During execution, logs will display:

- Local decisions
- Conflict detection
- Advice requests
- Recursive forwarding
- Loop closure events

### Fast Execution (simulation)

To start 3 nodes into a single host for fast simulation, run the following command in cmd from the root project directory:
```
python run_simulation.py
```

All parameters can be set up through the [simulator/\__main\__.py](https://github.com/sequincozes/CounselorNode/blob/main/simulator/__main__.py).

### Docker Execution

With Docker engine running, run following command in terminal from the root project directory to build the containers:

```
docker compose build
```

To run the containers and see the logs in terminal, run the command:

```
docker compose up
```

<a id="configuration"></a>
## ⚙️ Configuration

Each node is configured via:

```
config/peer_config.json
```

Configurable parameters include:

- Node ID
- IP and Port
- List of peers
- Enabled classifiers
- F1-score margin (DCS selection)
- Minimum performance threshold
- Outlier percentile

This enables reproducible and parameterized experiments.

---

<a id="pt"></a>
#### PT 🇧🇷 

## ✔️ Visão Geral

O CounselorNode é uma implementação totalmente descentralizada peer-to-peer (P2P) de um nó de uma Counselors Network (CN) para Sistemas Colaborativos de Detecção de Intrusão (IDS).

Cada nó realiza classificação local utilizando **Seleção Dinâmica de Classificadores (DCS)** e solicita conselhos aos pares apenas quando a decisão local apresenta conflitos.

A ferramenta implementa:

- Classificação local com clustering
- Troca recursiva de conselhos
- Detecção de fechamento de ciclo
- Instrumentação de eventos

---

## 📚 Índice

- [Arquitetura](#arquitetura-pt)
- [Ambiente de testes](#ambiente-testes-pt)
- [Requerimentos](#requerimentos-pt)
- [Instalação](#instalacao-pt)
- [Execução](#execucao-pt)
- [Configuração](#configuracao-pt)

---

<a id="arquitetura-pt"></a>
## 🏗️ Arquitetura

Estrutura principal:

```
config/            # Configurações do nó
core/              # Lógica principal e classificação
infrastructure/    # Comunicação e logs
run_node.py        # Execução via linha de comando
```

### Componentes principais

**ClassifierEngine**
- Clusterização K-Means
- Seleção Dinâmica de Classificadores (Decision Tree, KNN, SVM e Naive Bayes por padrão)
- Detecção de conflitos
- Detecção de outliers

**CounselorNode**
- Executa classificação local
- Aciona requisição recursiva de conselhos
- Mantém lista de nós já consultados para evitar ciclos
- Detecta fechamento de loop

**Networking Layer**
- Comunicação TCP socket
- Troca de requisição/resposta de conselhos
- Controle do encaminhamento recursivo

---

<a id="ambiente-testes-pt"></a>
## 🖥️ Ambiente de testes

A ferramenta foi testada nas seguintes configurações:

| Configuração | Ambiente I | Ambiente II |
|----------|----------------|----------------|
| Sistema Operacional | Windows 11 | Ubuntu 20.04 LTS |
| Processador | AMD Ryzen 7 5700X3D | Intel Core i5-10300H |
| RAM | 16 GB | 16 GB |
| Arquitetura | 64-bit | 64-bit |

---

<a id="requerimentos-pt"></a>
## 📝 Requerimentos

CounselorNode é implementado em Python.

| Dependência | Versão Recomendada |
|------------|--------------------|
| Python | 3.9+ |
| Docker (opcional) | 28.1.1+ |

As bibliotecas estão listadas em:

```
requirements.txt
```

---

<a id="instalacao-pt"></a>
## ⚙️ Instalação

Clone o repositório:

```
git clone https://github.com/sequincozes/CounselorNode.git
cd CounselorNode
```

Crie um ambiente virtual (opcional):

### Linux / macOS

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

<a id="execucao-pt"></a>
## ▶️ Execução

Durante a execução, os logs irão mostrar:

- Decisões locais
- Detecção de conflitos
- Requisições de conselhos
- Encaminhamento recursivo
- Eventos de fechamento de loop

### Execução Rápida (simulation)

Para executar 3 nós em um único terminal para simulação rápida, execute o seguinte comando no terminal a partir pasta raiz do diretório:

```
python run_simulation.py
```

Todos os parâmetros podem ser editados através de [simulator/\__main\__.py](https://github.com/sequincozes/CounselorNode/blob/main/simulator/__main__.py).

### Execução via Docker

Com a Docker engine rodando, execute o seguinte comando no terminal a partir da pasta raiz do diretório para a criação dos containers:

```
docker compose build
```

Para executar os containers e ver os logs no terminal, execute o comando:

```
docker compose up
```

<a id="configuracao-pt"></a>
## ⚙️ Configuração

Arquivo:

```
config/peer_config.json
```

Permite definir:

- Porta e IP
- Lista de peers
- Classificadores
- Limiares de decisão
