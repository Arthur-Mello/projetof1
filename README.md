# Relatório – Predição de Voltas e Estratégias de Pitstop na Fórmula 1

## 1. Problema
No automobilismo, especialmente na Fórmula 1, a definição de estratégias de corrida é crucial para o desempenho dos pilotos. Um dos pontos mais importantes é decidir **quando realizar pitstops e qual pneu utilizar**.  

Este projeto utiliza dados da corrida de **Monza 2024** obtidos via biblioteca **FastF1**, com o objetivo de:  
- Treinar um modelo de machine learning para **predizer a qualidade das voltas futuras** considerando características como número da volta, setores, tipo de pneu e tempo acumulado.  
- Simular **estratégias de pitstop** baseadas nas condições de corrida e nas probabilidades fornecidas pelo modelo.  

---

## 2. Preparação dos Dados
- Fonte: `FastF1` (dados de telemetria da corrida).  
- Dados extraídos por volta:  
  - Piloto (`Driver`)  
  - Número da volta (`LapNumber`)  
  - Tempo da volta (`LapTime`)  
  - Tempos dos três setores (`Sector1`, `Sector2`, `Sector3`)  
  - Tipo de pneu (`Compound`)  

### Pré-processamento
1. Conversão de tempos para segundos.  
2. Normalização dos setores por piloto (valores relativos à média individual).  
3. Codificação do tipo de pneu em variável numérica (`LabelEncoder`).  
4. Criação de atributos adicionais:  
   - **PrevCompoundCode**: pneu anterior.  
   - **LapsSincePit**: voltas desde o último pitstop.  
   - **CumulativeTime**: tempo acumulado na corrida.  

### Rótulo (target)
- Definição de **GoodLap**:  
  - 1 = volta melhor que a média do piloto.  
  - 0 = volta pior que a média.  

---

## 3. Modelo de Machine Learning
Foi utilizado um **MLPClassifier (rede neural multicamada)** da biblioteca Scikit-learn com os seguintes parâmetros:  
- Camadas escondidas: `(15, 15)`  
- Máx. iterações: `5000`  
- Regularização (`alpha`): `0.001`  
- Seed: `42`  

Para lidar com o **desbalanceamento das classes** (nem todas as voltas são boas), foi aplicado `class_weight="balanced"`.  

Os dados foram divididos em **treino (80%)** e **teste (20%)**, de forma estratificada.  

---

## 4. Experimentos
### 4.1 Predição de boas voltas
O modelo foi treinado para classificar voltas em **boas (1)** ou **ruins (0)**.  
- Acurácia obtida no conjunto de teste: **81,8%**  

### 4.2 Simulação de pitstops
- Considerados apenas pitstops após 40% da corrida.  
- Vida útil estimada dos pneus:  
  - Soft: 15 voltas  
  - Medium: 25 voltas  
  - Hard: 35 voltas  
- Condições de decisão:  
  - Se o piloto estava lento demais em relação ao carro da frente ou atrás.  
  - Testadas diferentes opções de pneus no modelo, escolhendo aquela com maior probabilidade de gerar uma boa volta.  

---

## 5. Resultados
1. **Predição de boas voltas**:  
   - O modelo conseguiu identificar boas e más voltas com **acurácia de 81,8%**.  
   - Isso indica que a rede neural conseguiu capturar padrões entre desgaste do pneu, número de voltas e desempenho do piloto.  

2. **Estratégias de pitstop sugeridas**:  

| Piloto | Volta | Pneu Sugerido |
|--------|-------|---------------|
| LEC    | 51    | MEDIUM        |
| PIA    | 51    | MEDIUM        |
| NOR    | 42    | MEDIUM        |
| SAI    | 38    | SOFT          |
| HAM    | 51    | HARD          |
| VER    | 51    | MEDIUM        |
| RUS    | 51    | SOFT          |
| PER    | 42    | MEDIUM        |
| ALB    | 51    | MEDIUM        |
| MAG    | 51    | HARD          |
| ALO    | 51    | SOFT          |
| COL    | 51    | HARD          |
| RIC    | 42    | MEDIUM        |
| OCO    | 51    | MEDIUM        |
| GAS    | 51    | MEDIUM        |
| BOT    | 51    | HARD          |
| HUL    | 51    | HARD          |
| ZHO    | 52    | SOFT          |
| STR    | 51    | SOFT          |

---

## 6. Conclusão
Este estudo mostra que é possível aplicar técnicas de **machine learning** em dados de Fórmula 1 para:  
- **Avaliar o desempenho por volta** (identificação de boas e más voltas).  
- **Apoiar a tomada de decisão estratégica** em pitstops, sugerindo o melhor pneu em determinados momentos da corrida.  

Embora simplificado, o modelo abre caminho para aplicações reais, como:  
- Ajustar estratégias em tempo real.  
- Incorporar variáveis adicionais (clima, safety car, tráfego).  
- Simular cenários de corrida completos.  
