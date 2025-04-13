import streamlit as st
import json
import os
from dqn_agent import DQNAgent
from environment import EcoEnv
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="EcoBrain+", layout="wide")

data_path = os.path.join("data", "sample_data.json")
if not os.path.exists(data_path):
    st.error("❌ Données manquantes : `data/sample_data.json`")
    st.stop()

with open(data_path) as f:
    data = json.load(f)

env = EcoEnv(data)
agent = DQNAgent(state_size=14, action_size=6)

if 'state' not in st.session_state:
    st.session_state.state = env.state
    st.session_state.history = []
    st.session_state.rewards = []

st.title("🌱 EcoBrain+")
st.markdown("Optimisez votre consommation énergétique et environnementale grâce à une IA prédictive éthique et transparente.")

with st.sidebar:
    st.header("⚙️ Paramètres")
    st.write("Sélectionnez vos préférences ou contraintes actuelles :")
    vegan = st.checkbox("Régime végétarien/végétalien")
    teletravail = st.checkbox("Télétravail aujourd'hui")
    transport = st.selectbox("Mode de transport principal", ["Voiture", "Transport en commun", "Vélo", "Marche"])
    chauffage = st.slider("Température de chauffage (°C)", 16, 24, 20)

state = st.session_state.state
action = agent.act(state)
label = env.action_labels[action]
justifications = env.action_explanations[action]

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("🧠 Conseil IA personnalisé")
    st.success(f"**{label}**")
    st.caption(justifications)
    if st.button("✅ Appliquer ce conseil"):
        next_state, reward = env.step(action)
        agent.remember(state, action, reward, next_state)
        agent.train()
        st.session_state.state = next_state
        st.session_state.history.append(label)
        st.session_state.rewards.append(reward)

with col2:
    st.metric("Score éco", round(sum(st.session_state.rewards), 2))
    st.metric("CO2 (kg)", round(state[8], 2))
    st.metric("Électricité (MWh)", round(state[0], 2))

if st.session_state.rewards:
    fig, ax = plt.subplots()
    ax.plot(st.session_state.rewards, marker='o', color='green')
    ax.set_title("Évolution des bénéfices environnementaux")
    st.pyplot(fig)

with st.expander("🔍 Détails des données régionales utilisées"):
    st.json(data)