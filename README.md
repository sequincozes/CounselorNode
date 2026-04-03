## ConselourNode: Uma Implementação P2P de uma Counselors Network para Sistemas Colaborativos de Detecção de Intrusão

O CounselorNode é uma implementação peer-to-peer (P2P) de uma Counselors Network (CN) que utiliza por padrão 3 nós para Sistemas Colaborativos de Detecção de Intrusão (IDS), ferramenta essa utilizada no trabalho Ataques de Envenenamento de Rótulos contra a Detecção de Zero-Day em Sistemas de Detecção de Intrusão Colaborativos. Cada nó realiza classificação local utilizando **Seleção Dinâmica de Classificadores (DCS)** e solicita conselhos aos pares apenas quando a decisão local apresenta conflitos. Para experimentos foi também implementada uma função de envenenamento de conselho com base na taxa de envenenamento (poison rate) definida.

A ferramenta implementa:

- Classificação local com clustering
- Troca recursiva de conselhos
- Detecção de fechamento de ciclo
- Instrumentação de eventos
- Envenenamento de conselhos

A base de dados utilizada consiste em amostras dos parâmetros físicos e cibernéticos de um Veículo Aéreo não Tripulado sob 4 tipos de ataques: Replay, Evil Twin, False Data Injection (FDI) e Denial of Service (DoS), além de parâmetros sob comportamento normal representados pela classe Benign.

---

## Organização do README

- [Selos Considerados](#selos)
- [Informações e Arquitetura](#arquitetura)
- [Dependências](#dependencias)
- [Preocupações com Segurança](#seguranca)
- [Instalação](#instalacao)
- [Teste Mínimo](#teste-minimo)
- [Experimentos](#experimentos)
- [Considerações Finais](#consideracoes)
- [LICENSE](#license)

---

<a id="selos"></a>
## Selos Considerados

Os autores sugerem a consideração dos seguintes selos no processo de avaliação:

- Artefatos Disponíveis (SeloD)
- Artefatos Funcionais (SeloF)
- Artefatos Sustentáveis (SeloS)
- Experimentos Reprodutíveis (SeloR)

---

<a id="arquitetura"></a>
## Informações e Arquitetura

Estrutura principal:

```
config/            # Configurações do nó
core/              # Lógica principal e classificação
infrastructure/    # Comunicação e geração de logs
data/              # Datasets para treinamento e teste
logs/              # Arquivos .csv com logs dos nós
simulator/         # Execução do simulador
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

<a id="dependencias"></a>
## Dependências

CounselorNode é implementado em Python.

| Dependência | Versão Recomendada |
|------------|--------------------|
| Python | 3.9+ |

As bibliotecas estão listadas em:

```
requirements.txt
```

---

<a id="seguranca"></a>
## Preocupações com segurança

CounselourNode executa os nós por padrão nas portas 5000, 5001 e 5002.

---

<a id="instalacao"></a>
## Instalação

Clone o repositório:

```
git clone https://github.com/sequincozes/CounselorNode.git
cd CounselorNode
```

Crie e ative um ambiente virtual (opcional, mas recomendado):

### Linux / macOS

[Opcional] Se o Python não estiver instalado, você pode executar:

```
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

Após instalar o Python, execute no terminal:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

[Opcional] Se o Python não estiver instalado, você pode executar:

```
winget install Python.Python.3
```

Após instalar o Python:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

<a id="teste-minimo"></a>
## Teste Mínimo

Para iniciar a simulação dos 3 nós em um único host, execute o seguinte comando no terminal (na raiz do projeto):

```
python simulator/__main__.py
```

O sistema irá classificar um dataset com o total de 1000 amostras. Os logs do sistema e resultados dos classificadores estarão disponíveis na pasta logs\.
Durante a execução, os logs irão exibir:
- Decisões locais  
- Detecção de conflitos  
- Solicitações de conselho  
- Encaminhamento recursivo  
- Eventos de fechamento de loop

É possível inserir mais nós, alterar bases de dados e parâmetros de classificação, ativar envenenamento de conselhos através das configurações no arquivo simulator/__main__.py. Alterações relacionadas a adição/remoção de nó realizadas no arquivo simulator/__main__.py também devem ser feitas em config/peer_config.json.

---

<a id="experimentos"></a>
## Experimentos

Os experimentos do artigo foram conduzidos em uma CN de 3 nós, definindo como objeto de estudo o Nó 1 e considerando 4 cenários distintos:
1. Nó 1 sem conexão com a CN (aconselhamento desativado)
2. Nó 1 com conexão a CN (aconselhamento ativado), sem envenenamento (poison_rate = 0)
3. Nó 1 com conexão a CN, taxa de envenenamento de 50% (poison_rate = 0.5)
4. Nó 1 com conexão a CN, taxa de envenenamento de 50% (poison_rate = 1)

As seções a seguir descrevem o passo a passo para a replicação dos cenários

### 1. Nó 1 sem conexão com a CN (aconselhamento desativado)

No arquivo simulator/__main__.py, a configuração dos nós estará disposta da seguinte forma (linha 162):

```
nodes_config = [
        ("127.0.0.1", 5000, {
            "train_eval_dataset_source": "data_sbrc2026/treino_no1.csv",
            "final_test_dataset_source": "data_sbrc2026/teste_no1.csv",
            "target_column": "class",
            "eval_size": 0.30,
            "clustering_n_clusters": 5,
            "f1_threshold": 0.05,
            "f1_min_required": 0.80
        }),
        ("127.0.0.1", 5001, {
            "train_eval_dataset_source": "data_sbrc2026/no2.csv",
            "final_test_dataset_source": "data_sbrc2026/no2.csv",
            "target_column": "class",
            "eval_size": 0.30,
            "clustering_n_clusters": 5,
            "f1_threshold": 0.05,
            "f1_min_required": 0.80,
            "outlier_enabled": True
        }),
        ("127.0.0.1", 5002, {
            "train_eval_dataset_source": "data_sbrc2026/no3.csv",
            "final_test_dataset_source": "data_sbrc2026/no3.csv",
            "target_column": "class",
            "eval_size": 0.30,
            "clustering_n_clusters": 5,
            "f1_threshold": 0.05,
            "f1_min_required": 0.80,
            "outlier_enabled": True
        }),
    ]
```
Remova os 2 últimos nós, deixando somente o primeiro nó (porta 5000):

```
nodes_config = [
        ("127.0.0.1", 5000, {
            "train_eval_dataset_source": "data_sbrc2026/treino_no1.csv",
            "final_test_dataset_source": "data_sbrc2026/teste_no1.csv",
            "target_column": "class",
            "eval_size": 0.30,
            "clustering_n_clusters": 5,
            "f1_threshold": 0.05,
            "f1_min_required": 0.80
        }),
    ]
```

No arquivo config/peer_config.json, a configuração dos nós estará disposta da seguinte forma (linha 5):

```
"counselor_peers": [
    {
      "name": "no_1",
      "ip": "127.0.0.1",
      "port": 5000
    },
    {
      "name": "no_2",
      "ip": "127.0.0.1",
      "port": 5001
    },
    {
      "name": "no_3",
      "ip": "127.0.0.1",
      "port": 5002
    }
  ],
```
Remova os 2 últimos nós, deixando somente o primeiro nó (porta 5000):

```
counselor_peers": [
    {
      "name": "no_1",
      "ip": "127.0.0.1",
      "port": 5000
    },
],
```

Para iniciar o simulador, execute o comando no terminal aberto na pasta raiz do projeto:

```
python simulator/__main__.py
```

As métricas podem ser calculadas a partir dos arquivos .csv gerados na pasta logs\.

### 2. Nó 1 com conexão a CN (aconselhamento ativado), sem envenenamento (poison_rate = 0)

Assim como no [Teste Mínimo](#teste-minimo), retorne os arquivos simulator/__main__.py e config/peer_config.json às configurações originais:

```
nodes_config = [
        ("127.0.0.1", 5000, {
            "train_eval_dataset_source": "data_sbrc2026/treino_no1.csv",
            "final_test_dataset_source": "data_sbrc2026/teste_no1.csv",
            "target_column": "class",
            "eval_size": 0.30,
            "clustering_n_clusters": 5,
            "f1_threshold": 0.05,
            "f1_min_required": 0.80
        }),
        ("127.0.0.1", 5001, {
            "train_eval_dataset_source": "data_sbrc2026/no2.csv",
            "final_test_dataset_source": "data_sbrc2026/no2.csv",
            "target_column": "class",
            "eval_size": 0.30,
            "clustering_n_clusters": 5,
            "f1_threshold": 0.05,
            "f1_min_required": 0.80,
            "outlier_enabled": True
        }),
        ("127.0.0.1", 5002, {
            "train_eval_dataset_source": "data_sbrc2026/no3.csv",
            "final_test_dataset_source": "data_sbrc2026/no3.csv",
            "target_column": "class",
            "eval_size": 0.30,
            "clustering_n_clusters": 5,
            "f1_threshold": 0.05,
            "f1_min_required": 0.80,
            "outlier_enabled": True
        }),
    ]
```
```
"counselor_peers": [
    {
      "name": "no_1",
      "ip": "127.0.0.1",
      "port": 5000
    },
    {
      "name": "no_2",
      "ip": "127.0.0.1",
      "port": 5001
    },
    {
      "name": "no_3",
      "ip": "127.0.0.1",
      "port": 5002
    }
  ],
```

Para iniciar o simulador, execute o comando no terminal aberto na pasta raiz do projeto:

```
python simulator/__main__.py
```

As métricas podem ser calculadas a partir dos arquivos .csv gerados na pasta logs\.

### 3. Nó 1 com conexão a CN, taxa de envenenamento de 50% (poison_rate = 0.5)

 No arquivo simulator/__main__.py, logo abaixo das configurações dos nós, altere o parâmetro poison_rate de 0.0 para 0.5 (linha 197):

```
poison_rate = 0.5
```

Para iniciar o simulador, execute o comando no terminal aberto na pasta raiz do projeto:

```
python simulator/__main__.py
```

As métricas podem ser calculadas a partir dos arquivos .csv gerados na pasta logs\.

### 4. Nó 1 com conexão a CN, taxa de envenenamento de 100% (poison_rate = 1)

 No arquivo simulator/__main__.py, logo abaixo das configurações dos nós, altere o parâmetro poison_rate de 0.5 para 1 (linha 197):

```
poison_rate = 1
```

Para iniciar o simulador, execute o comando no terminal aberto na pasta raiz do projeto:

```
python simulator/__main__.py
```

As métricas podem ser calculadas a partir dos arquivos .csv gerados na pasta logs\.

<a id="consideracoes"></a>
## Considerações Finais

As métricas e matrizes de confusão dos experimentos apresentados no artigo foram calculadas com o arquivo no_1_decisoes.csv gerado na pasta logs\ a partir da comparação da coluna "decisao" com a coluna "ground_truth", que exibem, respectivamente, a classe atribuida pelo sistema e a classe correta.

Os resultados obtidos nas execuções (principalmente cenário 3) podem não ser totalmente iguais aos obtidos nos experimentos do artigo, uma vez que a taxa de envenenamento (poison_rate) corresponde à chance de envenenamento do conselho e não à porcentagem de conselhos envenenados emitidos em relação ao número total.

<a id="license"></a>
## LICENSE

Este projeto está sob licença MIT. Consulte o arquivo LICENSE
