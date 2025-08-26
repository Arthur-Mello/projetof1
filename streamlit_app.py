import streamlit as st
import fastf1
import os

# garante que existe a pasta de cache
os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")

st.title("Análise de Corridas de F1 com FastF1 🚦")

# Seleção de ano e tipo de sessão
ano = st.sidebar.selectbox("Selecione o ano", list(range(2018, 2025))[::-1])
sessao_tipo = st.sidebar.selectbox("Selecione a sessão", ["FP1", "FP2", "FP3", "Q", "R"])

# Lista de corridas disponíveis no ano
corridas = fastf1.get_event_schedule(ano, include_testing=False)

# Seleciona o circuito direto
corrida = st.selectbox("Escolha o circuito", corridas["EventName"])
st.write(f"Você escolheu o circuito: **{corrida}**")

# Carrega a sessão escolhida
sessao = fastf1.get_session(ano, corrida, sessao_tipo)
sessao.load()

# Lista de pilotos
pilotos = sessao.drivers
pilotos_nomes = {pil: sessao.get_driver(pil)["FullName"] for pil in pilotos}

# Seleciona o piloto
piloto_escolhido = st.selectbox("Escolha o piloto", list(pilotos_nomes.values()))
st.write(f"Você selecionou: **{piloto_escolhido}**")

# Filtra voltas do piloto
cod_piloto = [k for k, v in pilotos_nomes.items() if v == piloto_escolhido][0]
voltas = sessao.laps.pick_drivers(cod_piloto).copy()

# Converte LapTime e setores para segundos
voltas["LapTimeSeconds"] = voltas["LapTime"].dt.total_seconds()
voltas["Sector1Seconds"] = voltas["Sector1Time"].dt.total_seconds()
voltas["Sector2Seconds"] = voltas["Sector2Time"].dt.total_seconds()
voltas["Sector3Seconds"] = voltas["Sector3Time"].dt.total_seconds()

# Mostra primeiras voltas com tempos em segundos
st.dataframe(
    voltas[["LapNumber", "LapTimeSeconds", "Sector1Seconds", "Sector2Seconds", "Sector3Seconds"]].head(10)
)