# 🌐 CounselorNode 🌐

## 🇺🇸 EN

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
- [Multi-Node Deployment Example](#multi-node-deployment-example)
- [🇧🇷 Versão em Português](#-versão-em-português)

---

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
- Dynamic Classifier Selection (Decision Tree, KNN, SVM by default)
- Conflict detection
- Outlier detection

**CounselorNode**
- Executes local classification
- Triggers recursive advice requests
- Handles forwarding-node tracking
- Detects loop closure

**Networking Layer**
- TCP socket-based communication
- Advice request/response exchange
- Recursive forwarding control

---

## 🖥️ Test Environment

The tool was tested under the following configurations:

| Setting | Environment I | Environment II |
|----------|----------------|----------------|
| OS | Windows 11 | Ubuntu 20.04 LTS |
| Processor | AMD Ryzen 7 5700X3D | Intel Core i5-10300H |
| RAM | 16 GB | 16 GB |
| Architecture | 64-bit | 64-bit |

---

## 📝 Requirements

CounselorNode is implemented in Python.

| Dependency | Recommended Version |
|------------|--------------------|
| Python | 3.9+ |

Libraries are listed in:

```
requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/sequincozes/CounselorNode.git
cd CounselorNode
```

Create and activate a virtual environment (optional):

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

## ▶️ Execution

To start a single node:

```
python -m counselornode.run_node 5000
```

Where `5000` is the TCP port used by the node.

Stop execution with:

```
Ctrl + C
```

---

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

## 🔗 Multi-Node Deployment Example

To simulate a Counselors Network locally, open three terminals:

```
Terminal 1:
python -m counselornode.run_node 5000

Terminal 2:
python -m counselornode.run_node 5001

Terminal 3:
python -m counselornode.run_node 5002
```

Ensure that each node’s `peer_config.json` includes the other peers.

During execution, logs will display:

- Local decisions
- Conflict detection
- Advice requests
- Recursive forwarding
- Loop closure events

---

# 🇧🇷 PT

## ✔️ Visão Geral

O CounselorNode é uma implementação totalmente descentralizada peer-to-peer (P2P) de um nó de uma Counselors Network (CN) para Sistemas Colaborativos de Detecção de Intrusão (IDS).

Cada nó realiza classificação local utilizando **Seleção Dinâmica de Classificadores (DCS)** e solicita conselhos aos pares apenas quando a decisão local apresenta conflitos.

A ferramenta implementa:

- Classificação local com clustering
- Troca recursiva de conselhos
- Detecção de fechamento de ciclo
- Instrumentação de eventos

---

## 🏗️ Arquitetura

Estrutura principal:

```
config/            # Configurações do nó
core/              # Lógica principal e classificação
infrastructure/    # Comunicação e logs
run_node.py        # Execução via linha de comando
```

---

### Componentes principais

**ClassifierEngine**
- Clusterização K-Means 
- Seleção Dinâmica de Classificadores (Decision Tree, KNN, SVM por padrão)
- Detecção de conflitos
- Detecção de outliers

**CounselorNode**
- Executa classificação local
- Aciona requisição recursiva de conselhos
- Gerencia a cadeia de encaminhamento dos conselhos
- Detecta fechamento de loops

**Networking Layer**
- Comunicação TCP socket
- Trocas de requisão/resposta de conselhos
- Controle do encaminhamento recursivo

---

## 🖥️ Ambiente de testes

A ferramenta foi testada nas seguintes configurações:

| Configuração | Ambiente I | Ambiente II |
|----------|----------------|----------------|
| Sistema Operacional | Windows 11 | Ubuntu 20.04 LTS |
| Processador | AMD Ryzen 7 5700X3D | Intel Core i5-10300H |
| RAM | 16 GB | 16 GB |
| Arquitetura | 64-bit | 64-bit |

---

## 📝 Requerimentos

CounselorNode é implementado em Python.

| Dependência | Versão Recomendada |
|------------|--------------------|
| Python | 3.9+ |

As bibliotecas estão listadas em:

```
requirements.txt
```

---

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

## ▶️ Execução

Para iniciar um nó:

```
python -m counselornode.run_node 5000
```

---

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

---

## 🔗 Execução com 3 Nós

Execute em três terminais diferentes:

```
python -m counselornode.run_node 5000
python -m counselornode.run_node 5001
python -m counselornode.run_node 5002
```

Certifique-se de que cada nó esteja configurado com os demais como peers.

Durante a execução, os logs irão mostrar:

- Decisões Locais
- Detecção de conflitos
- Requisições de conselhos
- Encaminhamento recursivo
- Eventos de fechamento de loop

---
