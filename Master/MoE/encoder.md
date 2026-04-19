# TimeSeriesEncoder +  Gating MLP

## Contexto

O encoder recebe `data_norm` — a série já normalizada (média≈0, std≈1) — com shape `(B, L)`,
onde B é o batch size e L é o comprimento do contexto (variável entre batches).
Sua única responsabilidade: produzir um embedding de tamanho fixo `(B, 64)`, independente de L.

---

## unsqueeze(1): (B, L) → (B, 1, L)

Insere uma dimensão de canal na posição 1. Os valores não mudam.

**Por quê:** `Conv1d` espera entrada no formato `(Batch, Canais, Comprimento)`.
Sem o unsqueeze, o tensor não tem dimensão de canal e o Conv1d não sabe como processar.

```
ANTES — shape (2, 168):
  Série A: [-0.20, -0.87,  0.30, -0.64,  0.06, ...,  1.24]   ← linha plana
  Série B: [ 0.19,  1.43, -0.33,  0.81, -0.43, ..., -0.90]

DEPOIS — shape (2, 1, 168):
  Série A: [[-0.20, -0.87,  0.30, -0.64,  0.06, ...,  1.24]]  ← dentro de [[]]
  Série B: [[ 0.19,  1.43, -0.33,  0.81, -0.43, ..., -0.90]]
                ↑
            canal 0 (único canal)
```

A diferença é puramente estrutural: antes era uma lista de 168 números, agora é
uma lista com 1 sublista de 168 números. Para o PyTorch isso muda como o dado é
interpretado nas convoluções.

---

## InstanceNorm1d(1, affine=True): (B, 1, L) → (B, 1, L)

Normaliza cada série individualmente sobre sua dimensão temporal L:

```
output[b, c, t] = (x[b, c, t] - mean_L) / std_L  ×  γ  +  β
                                                      ↑       ↑
                                               scale aprendível  shift aprendível
```

Como `data_norm` já chegou com média≈0 e std≈1 da normalização da Etapa 5,
o efeito numérico é mínimo. Os únicos parâmetros que atuam são γ e β (affine),
inicializados em 1 e 0.

```
ENTRADA canal 0, Série A:  [-0.20, -0.87,  0.30, -0.64, ...,  1.24]  mean≈0.01  std≈0.98
SAÍDA   canal 0, Série A:  [-0.21, -0.88,  0.30, -0.65, ...,  1.26]  ← quase idêntico
```

---

## Bottleneck Conv1d(1, 32, kernel_size=1) + GroupNorm + GELU
##       (B, 1, L) → (B, 32, L)

Uma convolução com kernel_size=1 olha **um timestep por vez**, de forma isolada.
Em cada posição t, pega o único valor do canal 0 e aplica 32 pesos aprendidos
diferentes, gerando 32 saídas distintas. É como aplicar 32 funções lineares ao
mesmo ponto.

```
ANTES — (2, 1, 168), primeiros 3 timesteps da Série A:
  t=0:  [ -0.21 ]   ← 1 valor
  t=1:  [ -0.88 ]
  t=2:  [  0.30 ]

DEPOIS — (2, 32, 168), primeiros 3 timesteps da Série A:
  t=0:  [ 0.04, -0.09,  0.15, -0.03,  0.22, -0.11,  0.07, ..., -0.06 ]  ← 32 valores
  t=1:  [ 0.18, -0.37,  0.62, -0.14,  0.91, -0.45,  0.29, ..., -0.26 ]  ← 32 valores
  t=2:  [-0.06,  0.13, -0.21,  0.05, -0.31,  0.15, -0.10, ...,  0.09 ]  ← 32 valores
```

Cada um dos 32 canais aprendeu a capturar um aspecto diferente do valor bruto.
O GroupNorm normaliza dentro de grupos de canais (estabilidade de treino).
O GELU adiciona não-linearidade.

---

## Branches paralelas (k=3, k=9, k=21): (B, 32, L) → 3× (B, 32, L)

Cada branch aplica `Conv1d(32, 32, kernel_size=k, padding='same')`.
O `padding='same'` com stride=1 garante que L não muda na saída.

A diferença entre os kernels é o "campo de visão" — quantos timesteps vizinhos
são combinados para gerar o valor em cada posição t:

```
Branch k=3  — janela de 3 timesteps (t-1, t, t+1):
  Saída em t=5 combina valores de t=4, t=5, t=6
  Captura: variações rápidas, ruído, picos pontuais

Branch k=9  — janela de 9 timesteps (t-4 até t+4):
  Saída em t=5 combina valores de t=1 até t=9
  Captura: tendências de curto prazo, sazonalidade semanal

Branch k=21 — janela de 21 timesteps (t-10 até t+10):
  Saída em t=5 combina valores de t=0 até ~t=15
  Captura: tendências longas, ciclos mensais
```

As 3 saídas têm shape `(2, 32, 168)` — L não muda em nenhuma delas.

---

## Skip branch: MaxPool1d + Conv1d(32, 32, k=1): (B, 32, L) → (B, 32, L)

`MaxPool1d(kernel_size=3, stride=1, padding=1)` em cada posição t retorna o
máximo entre os valores de t-1, t, t+1. Detecta picos locais em cada canal:

```
Canal 0 da Série A antes do MaxPool:
  [...,  0.04,  0.18, -0.06,  0.33, -0.21, ...]
          t=0    t=1    t=2    t=3    t=4

Após MaxPool (janela de 3):
  t=1: max(0.04,  0.18, -0.06) = 0.18
  t=2: max(0.18, -0.06,  0.33) = 0.33
  t=3: max(-0.06, 0.33, -0.21) = 0.33
```

O Conv1d(32,32,1) seguinte faz mixing de canais sobre os picos detectados.
Essa branch complementa as convoluções regulares que tendem a suavizar os extremos.

---

## torch.cat([k3, k9, k21, skip], dim=1): 4× (B, 32, L) → (B, 128, L)

Empilha os 4 branches na dimensão de canal. Cada timestep t passa a ter
128 valores descrevendo aquele ponto sob 4 perspectivas simultâneas:

```
Em t=5 da Série A — shape (128,) por timestep:
  Canais   0–31:  [ 0.12, -0.05, 0.31, ...]   ← branch k=3  (padrão local)
  Canais  32–63:  [ 0.08,  0.17, 0.22, ...]   ← branch k=9  (padrão médio)
  Canais  64–95:  [ 0.03,  0.09, 0.15, ...]   ← branch k=21 (padrão longo)
  Canais 96–127:  [ 0.18, -0.02, 0.41, ...]   ← skip (picos detectados)
```

---

## Global Average Pooling mean(dim=-1): (B, 128, L) → (B, 128)

**Este é o ponto-chave de toda a arquitetura.**

Para cada um dos 128 canais, calcula a média de todos os L timesteps.
A dimensão temporal colapsa completamente.

```
Canal 0 da Série A ao longo do tempo (L=168 valores):
  [ 0.12,  0.04, -0.06,  0.31,  0.08, ...,  0.19]
    t=0    t=1    t=2    t=3    t=4         t=167
  → média = 0.09

Canal 1 da Série A:
  [-0.05,  0.18,  0.22, -0.11,  0.07, ..., -0.03]
  → média = 0.03

Canal 127 da Série A:
  [ 0.41,  0.38,  0.45,  0.39,  0.42, ...,  0.44]
  → média = 0.41

Resultado da Série A — shape (128,):
  [0.09, 0.03, 0.17, -0.04, 0.22, ..., 0.41]
```

**Por que isso resolve o problema de tamanhos variáveis?**
A média não depende de quantos pontos existem. Uma série de L=168 e uma de L=336
produzem ambas um vetor de 128 números. O encoder não sabe nem precisa saber
qual era o comprimento original.

```
Série A (L=168): média de 168 pontos → vetor (128,)
Série C (L=336): média de 336 pontos → vetor (128,)  ← mesmo shape de saída
```

---

## MLP de projeção: (B, 128) → (B, 64)

```
Linear(128→64) → GELU → Dropout(0.1) → Linear(64→64)

ENTRADA por amostra — shape (128,):
  [0.09, 0.03, 0.17, -0.04, 0.22, ..., 0.41]

SAÍDA por amostra — shape (64,):   ← context_repr
  [0.23, -0.51, 0.18, 0.72, -0.09, ..., -0.34]
```

`context_repr (B, 64)` é o embedding final: um vetor de tamanho fixo que
resume a série inteira, independente do comprimento original.

---

## Gating MLP: (B, 64) → (B, 5)

`context_repr` é a **única entrada** do roteador. Os experts não participam aqui.

```
gating = Linear(64→32) + GELU + Linear(32→5)

ENTRADA: context_repr (2, 64)
  Série A: [0.23, -0.51, 0.18, 0.72, ..., -0.34]
  Série B: [-0.41, 0.29, -0.63, 0.15, ...,  0.57]

SAÍDA: logits (2, 5) — um score bruto por expert
  Série A: [ 1.20,  0.80, -0.30,  0.50, -0.90]
  Série B: [ 0.30,  1.50,  0.90, -0.20,  0.10]
             Moirai TimeMoE TimesFM Timer Chronos

→ softmax → probs_clean (2, 5):
  Série A: [0.38, 0.26, 0.08, 0.20, 0.08]
  Série B: [0.13, 0.44, 0.24, 0.08, 0.11]
```

O roteador aprendeu: "dado esse perfil de série (embedding), qual expert
historicamente produz melhores previsões".

---

## Resumo do fluxo completo

```
data_norm (B, L)
    │
    ├─ unsqueeze(1)           → (B, 1, L)       adiciona dim de canal
    ├─ InstanceNorm1d         → (B, 1, L)       normalização temporal por amostra
    ├─ Bottleneck Conv k=1    → (B, 32, L)      projeta 1→32 canais por timestep
    │
    ├─ Branch k=3             → (B, 32, L)      padrões locais (janela 3)
    ├─ Branch k=9             → (B, 32, L)      padrões médios (janela 9)
    ├─ Branch k=21            → (B, 32, L)      padrões longos (janela 21)
    ├─ Skip MaxPool+Conv      → (B, 32, L)      picos detectados
    │
    ├─ cat(dim=1)             → (B, 128, L)     4 perspectivas por timestep
    ├─ mean(dim=-1)  ← GAP   → (B, 128)        L some aqui — invariância de tamanho
    ├─ Linear+GELU+Drop+Lin   → (B, 64)         context_repr (embedding fixo)
    │
    └─ Gating MLP             → (B, 5)          logits → softmax → pesos dos experts
```

---

## O que cada parte recebe

| Componente      | Recebe              | Produz           |
|-----------------|---------------------|------------------|
| Encoder         | data_norm (B, L)    | context_repr (B, 64) |
| Gating MLP      | context_repr (B, 64)| logits (B, 5)    |
| Experts         | data_norm (B, L)    | previsões (B, H) |

Os experts recebem `data_norm` — a mesma série normalizada que entrou no encoder,
não o embedding. Eles fazem a previsão de forma independente, sem ver o roteador.
