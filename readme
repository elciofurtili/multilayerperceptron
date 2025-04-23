# Atividade de Redes Neurais Artificiais – Rede MLP

Este repositório contém a implementação completa da Atividade 3 da disciplina de Redes Neurais Artificiais do Programa de Pós-Graduação em Ciência da Computação (PPGCC-UNESP).

O objetivo principal é implementar uma rede neural do tipo Multi-Layer Perceptron (MLP) utilizando PyTorch, de forma manual (sem o uso de modelos prontos), para resolver problemas de classificação com diferentes níveis de complexidade, explorando diferentes arquiteturas, funções de ativação, estratégias de regularização e otimizadores.

---

## 🔧 Estrutura do Projeto

.
├── src/
│   ├── mlp.py        # Implementação para o Experimento 3.1 (Dataset 2)
│   └── mlp2.py       # Implementação para o Experimento 3.2 (Dataset 4)
├── dataset/
│   ├── 2/            # Dataset para o experimento 3.1 (problema não-linear)
│   │   └── ...       # Arquivos do Dataset 2
│   └── 4/            # Dataset para o experimento 3.2 (50 features, 4 classes)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── README.md

---

## ✅ Atendendo aos Requisitos da Atividade

O projeto foi desenvolvido para atender integralmente aos requisitos especificados no PDF da atividade:

### 1. Implementação Manual da MLP
- Uso da classe nn.Module do PyTorch
- Arquitetura flexível: permite definir o número de camadas e neurônios por camada
- Suporte às funções de ativação ReLU e Tanh

### 2. Treinamento e Validação
- Implementação do treinamento com SGD e Adam
- Uso de CrossEntropyLoss (multiclasse) e BCEWithLogitsLoss (binário, apenas no 3.1)
- Registro das métricas: perda e acurácia
- Early Stopping com paciência configurável
- Avaliação no conjunto de teste

### 3. Experimentos

#### 🔹 Experimento 3.1 – Dataset 2 (src/mlp.py)
- Problema não-linear com duas classes
- Arquitetura pequena e visualização das fronteiras de decisão
- Foco em resolver a não-linearidade com camadas ocultas

#### 🔹 Experimento 3.2 – Dataset 4 (src/mlp2.py)
- Execução automatizada de todas as combinações de:
  - Arquiteturas (pequena, média, grande)
  - Otimizadores (SGD, Adam)
  - Ativações (ReLU, Tanh)
  - Regularização (nenhuma, Dropout, L2)
- Resultados organizados para comparação
- Avaliação final no conjunto de teste

---

## 📊 Resultados
Os experimentos realizados mostraram que:
- Redes maiores com regularização L2 tendem a apresentar o melhor desempenho.
- O otimizador Adam teve melhor estabilidade em arquiteturas mais profundas.
- A função de ativação ReLU foi mais eficiente em redes maiores.

---

## 🚀 Como Executar

1. Clone o repositório:
   git clone https://github.com/seu-usuario/nome-do-repo.git
   cd nome-do-repo

2. Instale as dependências:
   pip install torch pandas matplotlib numpy

3. Execute os scripts conforme a atividade desejada:
   python src/mlp.py     # Para o Experimento 3.1
   python src/mlp2.py    # Para o Experimento 3.2