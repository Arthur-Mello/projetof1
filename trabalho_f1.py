import os
import pandas as pd
import fastf1
from fastf1 import plotting
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Criar cache
cache_dir = os.path.join(os.getcwd(), "f1_cache")
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache('f1_cache')

# Carregar sessão
session = fastf1.get_session(2024, 'Spa', 'R')  # Corrida
session.load()
laps = session.laps

# Função para converter tempo
def to_seconds(td):
    if pd.isnull(td):
        return None
    return td.total_seconds()

# Dataset
df = pd.DataFrame({
    'Driver': laps['Driver'],
    'LapNumber': laps['LapNumber'],
    'LapTime': laps['LapTime'].apply(to_seconds),
    'Sector1': laps['Sector1Time'].apply(to_seconds),
    'Sector2': laps['Sector2Time'].apply(to_seconds),
    'Sector3': laps['Sector3Time'].apply(to_seconds),
    'Compound': laps['Compound'].str.upper()
}).dropna()

# Codificação de pneus
le = LabelEncoder()
df['CompoundCode'] = le.fit_transform(df['Compound'])

# Criar rótulo: volta boa (melhor que média do piloto) ou ruim
df['GoodLap'] = df.groupby('Driver')['LapTime'].transform(lambda x: (x < x.mean()).astype(int))

# Features e rótulo
X = df[['LapNumber', 'Sector1', 'Sector2', 'Sector3', 'CompoundCode']]
y = df['GoodLap']

# Treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# Treinar MLP
mlp = MLPClassifier(hidden_layer_sizes=(25, 25), max_iter=150000, random_state=42)
mlp.fit(X_train, y_train)

print("📊 Acurácia:", accuracy_score(y_test, mlp.predict(X_test)))

# -----------------------------
# SIMULAÇÃO DE PITS
# -----------------------------

# Total de voltas da corrida
total_voltas = int(df['LapNumber'].max())
print(f"🏁 Total de voltas da corrida: {total_voltas}")

# -----------------------------
# SIMULAÇÃO DE PITS
# -----------------------------
recomendacoes = []

for driver in df['Driver'].unique():
    driver_laps = df[df['Driver'] == driver]

    # escolhe uma volta intermediária como ponto de simulação
    lap = int(driver_laps['LapNumber'].median())
    ultima_volta = driver_laps[driver_laps['LapNumber'] == lap].iloc[0]

    melhores = []
    for comp in le.classes_:
        X_pneu = pd.DataFrame({
            'LapNumber': [lap+1],
            'Sector1': [ultima_volta['Sector1']],
            'Sector2': [ultima_volta['Sector2']],
            'Sector3': [ultima_volta['Sector3']],
            'CompoundCode': [le.transform([comp])[0]]
        })
        pred_novo = mlp.predict(X_pneu)[0]
        melhores.append((comp, pred_novo))

    # pega o composto que maximiza a chance de volta boa
    best_comp, best_pred = max(melhores, key=lambda x: x[1])

    strategies = []
    recomendacoes.append((
        driver,
        lap,
        best_comp,
        "Boa" if best_pred else "Ruim",
        len(strategies)
    ))

# Limiar de perda para sugerir pitstop (em segundos)
LIMIAR_FRENTE = 1.5
LIMIAR_TRAS = 1.0

for driver in df['Driver'].unique():
    driver_laps = df[df['Driver'] == driver].sort_values('LapNumber').reset_index(drop=True)

    # pega a primeira volta existente
    primeira_volta = driver_laps.iloc[0]
    current_compound = primeira_volta['Compound']
    lap = primeira_volta['LapNumber']

    while lap <= total_voltas:
        # Última volta registrada do piloto até "lap"
        if lap not in driver_laps['LapNumber'].values:
            lap += 1
            continue
        ultima_volta = driver_laps[driver_laps['LapNumber'] == lap].iloc[0]
        tempo_piloto = ultima_volta['Sector1'] + ultima_volta['Sector2'] + ultima_volta['Sector3']

        # pega tempos da mesma volta de todos os pilotos
        todos_na_volta = df[df['LapNumber'] == lap].copy()
        todos_na_volta['LapTime'] = todos_na_volta['Sector1'] + todos_na_volta['Sector2'] + todos_na_volta['Sector3']

        # piloto da frente (tempo menor)
        frente = todos_na_volta[todos_na_volta['Driver'] != driver]['LapTime'].min()
        # piloto de trás (tempo maior)
        tras = todos_na_volta[todos_na_volta['Driver'] != driver]['LapTime'].max()

        # diferença de tempo
        diff_frente = tempo_piloto - frente if not pd.isna(frente) else 0
        diff_tras = tras - tempo_piloto if not pd.isna(tras) else 0

        # Checa se deve parar pelo desgaste
        if diff_frente > LIMIAR_FRENTE or diff_tras > LIMIAR_TRAS:
            # Simula escolha do melhor composto
            melhores = []
            for comp in le.classes_:
                X_pneu = pd.DataFrame({
                    'LapNumber': [lap],
                    'Sector1': [ultima_volta['Sector1']],
                    'Sector2': [ultima_volta['Sector2']],
                    'Sector3': [ultima_volta['Sector3']],
                    'CompoundCode': [le.transform([comp])[0]]
                })
                pred_novo = mlp.predict(X_pneu)[0]
                melhores.append((comp, pred_novo))

            best_comp, best_pred = max(melhores, key=lambda x: x[1])

            if best_comp != current_compound:
                strategies.append((driver, lap, best_comp))
                current_compound = best_comp

        lap += 1

# Resultado final em tabela
df_recomendacoes = pd.DataFrame(
    recomendacoes,
    columns=["Piloto", "Volta Recomendada", "Pneu Sugerido", "Expectativa", "Pitstops Estimados"]
)

print("\n🔧 Estratégia sugerida de pitstops:")
print(df_recomendacoes)
