import os
import pandas as pd
import fastf1
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
import numpy as np

# ================================
# Cache FastF1
# ================================
cache_dir = os.path.join(os.getcwd(), "f1_cache")
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache('f1_cache')

# ================================
# Carregar sessão
# ================================
session = fastf1.get_session(2024, 'Monza', 'R')  # Corrida
session.load()
laps = session.laps

# ================================
# Função para converter tempo
# ================================
def to_seconds(td):
    if pd.isnull(td):
        return None
    return td.total_seconds()

# ================================
# Criar DataFrame
# ================================
df = pd.DataFrame({
    'Driver': laps['Driver'],
    'LapNumber': laps['LapNumber'],
    'LapTime': laps['LapTime'].apply(to_seconds),
    'Sector1': laps['Sector1Time'].apply(to_seconds),
    'Sector2': laps['Sector2Time'].apply(to_seconds),
    'Sector3': laps['Sector3Time'].apply(to_seconds),
    'Compound': laps['Compound'].str.upper()
}).dropna()

# ================================
# Codificação de pneus
# ================================
le = LabelEncoder()
df['CompoundCode'] = le.fit_transform(df['Compound'])

# ================================
# Criar rótulo: boa volta ou ruim
# ================================
df['GoodLap'] = df.groupby('Driver')['LapTime'].transform(lambda x: (x < x.mean()).astype(int))

# ================================
# Normalizar setores por piloto
# ================================
for col in ['Sector1', 'Sector2', 'Sector3']:
    df[col + '_rel'] = df.groupby('Driver')[col].transform(lambda x: x - x.mean())

# ================================
# Features adicionais
# ================================
df['PrevCompoundCode'] = df.groupby('Driver')['CompoundCode'].shift(1).fillna(0)
df['LapsSincePit'] = df.groupby('Driver')['LapNumber'].diff().fillna(1)
df['CumulativeTime'] = df.groupby('Driver')['LapTime'].cumsum()

# ================================
# Features e rótulo
# ================================
feature_cols = ['LapNumber', 'Sector1_rel', 'Sector2_rel', 'Sector3_rel',
                'CompoundCode', 'PrevCompoundCode', 'LapsSincePit', 'CumulativeTime']
X = df[feature_cols]
y = df['GoodLap']

# ================================
# Balanceamento de classes
# ================================
classes = np.unique(y)
weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
class_weights_dict = dict(zip(classes, weights))

# ================================
# Treino/teste
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# Treinar MLP
# ================================
mlp = MLPClassifier(
    hidden_layer_sizes=(15, 15),
    max_iter=5000,
    alpha=0.001,
    random_state=42
)
mlp.fit(X_train, y_train)

# Avaliar modelo
y_pred = mlp.predict(X_test)
print("📊 Acurácia:", accuracy_score(y_test, y_pred))

# ================================
# Simulação de pitstops (1 por piloto)
# ================================
LIMIAR_FRENTE = 5.0
LIMIAR_TRAS = 10.0
strategies = []

total_voltas = int(df['LapNumber'].max())
VOLTA_MINIMA = int(total_voltas * 0.4)  # só considerar pitstop após 40% da corrida

# vida útil estimada dos pneus (exemplo)
durabilidade = {"SOFT": 15, "MEDIUM": 25, "HARD": 35}

for driver in df['Driver'].unique():
    driver_laps = df[df['Driver'] == driver].sort_values('LapNumber').reset_index(drop=True)
    current_compound = driver_laps.iloc[0]['Compound']
    pitstop_done = False
    chosen_strategy = None

    for idx, row in driver_laps.iterrows():
        lap = row['LapNumber']

        # só considerar pitstop após a volta mínima
        if lap < VOLTA_MINIMA:
            continue

        # opcional: só considerar se já rodou próximo da durabilidade do pneu
        if lap < durabilidade.get(current_compound, 20) * 0.7:
            continue

        tempo_piloto = row['Sector1'] + row['Sector2'] + row['Sector3']

        todos_na_volta = df[df['LapNumber'] == lap].copy()
        todos_na_volta['LapTime'] = todos_na_volta['Sector1'] + todos_na_volta['Sector2'] + todos_na_volta['Sector3']

        frente = todos_na_volta[todos_na_volta['Driver'] != driver]['LapTime'].min()
        tras = todos_na_volta[todos_na_volta['Driver'] != driver]['LapTime'].max()

        diff_frente = tempo_piloto - frente if not pd.isna(frente) else 0
        diff_tras = tras - tempo_piloto if not pd.isna(tras) else 0

        if diff_frente > LIMIAR_FRENTE or diff_tras > LIMIAR_TRAS:
            melhores = []
            for comp in le.classes_:
                X_pneu = pd.DataFrame({
                    'LapNumber': [lap],
                    'Sector1_rel': [row['Sector1'] - driver_laps['Sector1'].mean()],
                    'Sector2_rel': [row['Sector2'] - driver_laps['Sector2'].mean()],
                    'Sector3_rel': [row['Sector3'] - driver_laps['Sector3'].mean()],
                    'CompoundCode': [le.transform([comp])[0]],
                    'PrevCompoundCode': [row['PrevCompoundCode']],
                    'LapsSincePit': [row['LapsSincePit']],
                    'CumulativeTime': [row['CumulativeTime']]
                })
                prob = mlp.predict_proba(X_pneu)[0][1]
                melhores.append((comp, prob))

            comps = [comp for comp, prob in melhores]
            probs = np.array([prob for comp, prob in melhores])
            probs /= probs.sum()
            best_comp = np.random.choice(comps, p=probs)
            best_prob = dict(melhores)[best_comp]

            if best_comp != current_compound and best_prob > 0.5:
                chosen_strategy = (driver, int(lap), best_comp)
                pitstop_done = True
                current_compound = best_comp

    if chosen_strategy:
        strategies.append(chosen_strategy)

# Resultado final
df_strategies = pd.DataFrame(strategies, columns=["Piloto", "Volta", "Pneu Sugerido"])
print("\n🔧 Estratégia sugerida de pitstops (1 por piloto):")
print(df_strategies)