# 🔎 Pipeline MoE para Séries Temporais com Experts Fundacionais

---

## **1. Dataset (.jsonl)**

### **Entrada esperada**

* Arquivo `.jsonl`, cada linha contém:

```json
{"sequence": [9388.163, 4634.727, 2431.101, ..., 19713.733, 16773.048]}
```

* Campo obrigatório: `"sequence"`, que deve ser **lista de floats** (valores da série temporal).
* Pode ter várias séries, uma por linha.

### **Saída do `load_jsonl`**

* Uma lista Python de listas de floats:

```python
[
  [9388.163, 4634.727, 2431.101, ..., 19713.733, 16773.048],
  [1174.117, 648.583, 434.192, ..., 0.0, 0.0],
  ...
]
```

---

## **2. Dataset Loader (`TimeSeriesDataset`)**

### **Entrada esperada**

* `sequences`: lista de séries (list\[list\[float]])
* `input_length`: número de pontos usados como entrada do modelo.
* `pred_length`: número de pontos a serem previstos.

### **Processamento**

* Para cada série com tamanho ≥ `input_length + pred_length`, cria um par:

  * `inp = seq[:input_length]` → janela de entrada
  * `tgt = seq[input_length : input_length + pred_length]` → janela de saída esperada

### **Saída**

* Dataset que retorna tuplas:

```python
(inp: torch.FloatTensor, tgt: torch.FloatTensor)
```

onde:

* `inp.shape = (input_length,)`
* `tgt.shape = (pred_length,)`

---

## **3. Experts (`MoiraiMoEExpert`, `TimeMoEExpert`, `TimesFMExpert`)**

### **Entrada esperada (no `forward`)**

* Tensor de shape `(B, input_length)` (batch de séries).

### **Saída esperada**

* Tensor de shape `(B, pred_length)` com as predições para cada série do batch.

### **Observação importante**

* Cada expert já vem **pré-treinado** (zero-shot).
* Eles **não são treinados** aqui, apenas executados em `torch.no_grad()`.

---

## **4. Router (`MoERouter`)**

### **Entradas**

* `x: torch.Tensor` com shape `(B, input_length)` → batch de séries.

### **Etapas internas**

1. **Gating (linear + softmax):**

   * `logits = self.gating(x)` → `(B, n_experts)`
   * `probs = softmax(logits)` → `(B, n_experts)`
2. **Top-k routing (k=2 fixo):**

   * `topk_vals, topk_idx = torch.topk(probs, k=2, dim=-1)`
   * Para cada amostra, seleciona 2 experts com maiores probabilidades.
3. **Executa apenas os experts selecionados:**

   * Agrupa inputs por expert
   * Executa cada expert em `torch.no_grad()`
   * Junta resultados no tensor `preds_by_expert`
4. **Combinação ponderada:**

   * Para cada amostra `i`:

     * Normaliza pesos `weights = vals / vals.sum()`
     * Combina outputs: `final_pred[i] = Σ (weight_j * expert_out_j)`

### **Saída**

* Tensor `(B, pred_length)` com as previsões finais.

---

## **5. Treinamento (`train_and_save`)**

### **Entradas**

* `jsonl_path`: caminho do dataset `.jsonl`
* `input_length`: número de pontos de entrada
* `pred_length`: número de pontos de previsão
* `save_path`: caminho para salvar o modelo (`.pt`)
* `device`: `"cpu"` ou `"cuda"`
* Hiperparâmetros internos:

  * `batch_size=8`
  * `epochs=5`
  * `lr=1e-3`

### **Processamento**

1. Carrega dataset `.jsonl`
2. Cria `DataLoader`
3. Instancia `MoERouter`
4. Otimiza **apenas o gating** (`nn.Linear`)

   * Função de perda: `nn.HuberLoss(delta=2.0)`
   * Experts são **congelados** (zero-shot)

### **Saída**

* Modelo treinado salvo em `save_path`
* Arquivo contém:

```python
{
 "gating_state": state_dict,
 "expert_keys": ["moirai", "timemoe", "timesfm"],
 "pred_length": pred_length,
 "input_length": input_length
}
```

---

## **6. Predição (`predict_from_model`)**

### **Entradas**

* `model_path`: caminho do modelo salvo
* `series`: lista de floats (série completa ou parcial)
* `input_length`: número de pontos de entrada
* `device`: `"cpu"` ou `"cuda"`

### **Processamento**

1. Carrega `MoERouter` com pesos do gating.
2. Extrai últimos `input_length` pontos da série:

   ```python
   x = torch.tensor(series[-input_length:]).unsqueeze(0)  # (1, input_length)
   ```
3. Passa pelo roteador + experts.
4. Combina saída.

### **Saída**

* Tensor `(1, pred_length)` com previsões.

---

# 🔄 Resumo (fluxo completo)

1. **Dataset (`.jsonl`)**

   * Entrada: `{"sequence": [valores]}`
   * Saída: lista de séries

2. **Dataset Loader**

   * Entrada: séries, `input_length`, `pred_length`
   * Saída: pares `(inp, tgt)` como tensores

3. **Experts**

   * Entrada: `(B, input_length)`
   * Saída: `(B, pred_length)`

4. **Router**

   * Entrada: `(B, input_length)`
   * Saída: `(B, pred_length)` (fusão top-2)

5. **Treinamento**

   * Entrada: `.jsonl`, `input_length`, `pred_length`
   * Saída: modelo salvo (`.pt`)

6. **Predição**

   * Entrada: modelo salvo + série
   * Saída: previsões `(1, pred_length)`

---

# Exemplo: 


## **1. Dataset (.jsonl)**

### **Formato esperado**

Cada linha contém uma série temporal:

```json
{"sequence": [9388.163, 4634.727, 2431.101, 5521.988, 19713.733, 16773.048]}
{"sequence": [1174.117, 648.583, 434.192, 289.22, 0.0, 0.0]}
```

### **Entrada**

Arquivo `dados.jsonl` (2 séries no exemplo acima).

### **Saída de `load_jsonl`**

```python
[
  [9388.163, 4634.727, 2431.101, 5521.988, 19713.733, 16773.048],
  [1174.117, 648.583, 434.192, 289.22, 0.0, 0.0]
]
```

---

## **2. Dataset Loader (`TimeSeriesDataset`)**

### **Exemplo**

```python
ds = TimeSeriesDataset(sequences, input_length=4, pred_length=2)
```

### **Entrada**

* Série: `[9388.163, 4634.727, 2431.101, 5521.988, 19713.733, 16773.048]`
* `input_length = 4`
* `pred_length = 2`

### **Saída**

Um par `(inp, tgt)`:

```python
inp = tensor([9388.163, 4634.727, 2431.101, 5521.988])   # shape (4,)
tgt = tensor([19713.733, 16773.048])                    # shape (2,)
```

---

## **3. Experts (fundacionais: Moirai, TimeMoE, TimesFM)**

Cada expert segue a mesma interface.

### **Exemplo de uso**

```python
expert = TimeMoEExpert(prediction_length=2, device="cpu")
inp = torch.randn(3, 4)  # batch com 3 séries, cada uma com 4 pontos
out = expert(inp)
```

### **Entrada**

* Tensor `(B, input_length)`.
  Exemplo:

```python
tensor([
  [0.1, 0.2, 0.3, 0.4],
  [1.1, 1.2, 1.3, 1.4],
  [2.1, 2.2, 2.3, 2.4]
])
```

### **Saída**

* Tensor `(B, pred_length)`.
  Exemplo:

```python
tensor([
  [0.55, 0.66],
  [1.75, 1.88],
  [2.95, 3.04]
])
```

*(valores ilustrativos, cada expert retorna valores diferentes pois usam foundation models pré-treinados)*

---

## **4. Router (`MoERouter`)**

### **Exemplo de uso**

```python
router = MoERouter(input_length=4, pred_length=2, device="cpu")
x = torch.randn(2, 4)  # batch com 2 séries
out = router(x, verbose=True)
```

### **Entrada**

* Tensor `(B, input_length)`
  Exemplo:

```python
tensor([
  [0.5, 0.6, 0.7, 0.8],
  [1.5, 1.6, 1.7, 1.8]
])
```

### **Processo**

1. Calcula probabilidades por expert (softmax).
   Exemplo (3 experts):

   ```python
   probs = [[0.1, 0.7, 0.2],
            [0.4, 0.3, 0.3]]
   ```
2. Seleciona **top-2 experts** por amostra:

   * Série 1 → experts `[1, 2]` com pesos `[0.78, 0.22]`
   * Série 2 → experts `[0, 1]` com pesos `[0.57, 0.43]`
3. Executa apenas esses experts e combina saídas.

### **Saída**

* Tensor `(B, pred_length)`
  Exemplo:

```python
tensor([
  [10.3, 11.1],
  [ 9.7, 10.2]
])
```

---

## **5. Treinamento (`train_and_save`)**

### **Exemplo de uso**

```bash
python main.py --mode train --jsonl dados.jsonl --input-length 4 --pred-length 2 --save-path ./moe_model.pt --device cpu
```

### **Entrada**

* `jsonl_path="dados.jsonl"`
* `input_length=4`
* `pred_length=2`
* `save_path="./moe_model.pt"`

### **Processo**

1. Carrega dataset e gera `(inp, tgt)`
2. Forward → roteador + experts → predição `(B, pred_length)`
3. Calcula perda Huber entre predição e `tgt`
4. Atualiza apenas os pesos do **gating**
5. Repete por `epochs`

### **Saída**

Modelo salvo:

```python
{
 "gating_state": {...},        # pesos do roteador
 "expert_keys": ["moirai", "timemoe", "timesfm"],
 "pred_length": 2,
 "input_length": 4
}
```

Arquivo: `moe_model.pt`

---

## **6. Predição (`predict_from_model`)**

### **Exemplo de uso**

```python
series = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
out = predict_from_model("./moe_model.pt", series, input_length=4, device="cpu", verbose=True)
print(out)
```

### **Entrada**

* Série: `[0.5, 0.6, 0.7, 0.8, 1.0, 1.2]`
* O modelo usa apenas os últimos `input_length=4`:

```python
[0.7, 0.8, 1.0, 1.2]   # shape (1,4)
```

### **Processo**

* Roteador calcula probabilidades → seleciona top-2 experts → executa → combina outputs.

### **Saída**

* Tensor `(1, pred_length)`
  Exemplo:

```python
tensor([[1.35, 1.48]])
```

---

# 🔄 Fluxo Completo (resumido com exemplos)

1. **Dataset**

   * Input: `{"sequence": [9388.163, 4634.727, ...]}`
   * Output: `[[9388.163, 4634.727, ...], [1174.117, 648.583, ...]]`

2. **Dataset Loader**

   * Input: série + `input_length=4`, `pred_length=2`
   * Output: `(inp=[9388.163,4634.727,2431.101,5521.988], tgt=[19713.733,16773.048])`

3. **Experts**

   * Input: `(B=3, input_length=4)`
   * Output: `(B=3, pred_length=2)`

4. **Router**

   * Input: `(B=2, input_length=4)`
   * Output: `(B=2, pred_length=2)` (fusão top-2)

5. **Treinamento**

   * Input: `jsonl + input_length + pred_length`
   * Output: modelo salvo `.pt`

6. **Predição**

   * Input: `series + modelo salvo`
   * Output: `(1, pred_length)` com previsões