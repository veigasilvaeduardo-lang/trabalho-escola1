import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
import numpy as np

# Configuração da página
st.set_page_config(page_title="Painel Técnico de Vôlei", layout="wide")

# Função para carregar os dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv("tecnico_volei.csv", sep=";")
    df.columns = df.columns.str.strip().str.lower()
    return df

# Carregar os dados
df = carregar_dados()

# Título
st.title("🏐 Painel Técnico de Vôlei")

# Cards de métricas
media_idade = df["idade"].mean()
media_altura = df["altura_cm"].mean()
total_partidas = df["partidas_jogadas"].sum()
total_pontos = df["pontos_marcados"].sum()

st.markdown("## Estatísticas Gerais")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Média de Idade", f"{media_idade:.1f} anos")
with col2:
    st.metric("Altura Média", f"{media_altura:.1f} cm")
with col3:
    st.metric("Total de Partidas", total_partidas)
with col4:
    st.metric("Total de Pontos", total_pontos)

st.divider()

# Linha de gráficos
with st.container():
    col_graf1, col_graf2 = st.columns(2)

    with col_graf1:
        st.markdown("### Jogadores por Posição")
        fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
        sns.countplot(data=df, x="posicao", ax=ax1, palette="coolwarm")
        ax1.set_xlabel("Posição")
        ax1.set_ylabel("Qtd")
        ax1.tick_params(rotation=45)
        st.pyplot(fig)

    with col_graf2:
        st.markdown("### Jogadores por Turno de Treino")
        fig, ax2 = plt.subplots(figsize=(3.5, 2.5))
        sns.countplot(data=df, x="turno_treino", order=df["turno_treino"].value_counts().index, ax=ax2, palette="viridis")
        ax2.set_xlabel("Turno")
        ax2.set_ylabel("Qtd")
        st.pyplot(fig)

# Segunda linha de gráficos
with st.container():
    col_graf3, col_graf4 = st.columns(2)

    with col_graf3:
        st.markdown("### Distribuição de Altura por Posição")
        fig, ax3 = plt.subplots(figsize=(3.5, 2.5))
        sns.boxplot(data=df, x="posicao", y="altura_cm", ax=ax3, palette="pastel")
        ax3.set_xlabel("Posição")
        ax3.set_ylabel("Altura (cm)")
        st.pyplot(fig)

    with col_graf4:
        st.markdown("### Resultado do Último Jogo")
        fig, ax4 = plt.subplots(figsize=(3.5, 2.5))
        sns.countplot(data=df, x="resultado_ultimo_jogo", ax=ax4, palette="Set2")
        ax4.set_xlabel("Resultado")
        ax4.set_ylabel("Qtd")
        st.pyplot(fig)

st.divider()

# Exportar CSV
st.markdown("## Exportar Dados")
csv = df.to_csv(index=False, sep=';', encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    label="Baixar CSV",
    data=csv,
    file_name="tecnico_volei_export.csv",
    mime="text/csv",
)

st.divider()

# Análises estatísticas interativas
st.markdown("## Análises Estatísticas (Distribuições)")

# Binomial – probabilidade de vitória
st.markdown("### Probabilidade de Vitórias (Distribuição Binomial)")
p_vitoria = (df["resultado_ultimo_jogo"] == "Vitória").mean()

col_a, col_b = st.columns(2)
with col_a:
    n = st.slider("Número de jogos", min_value=1, max_value=50, value=10)
    k = st.slider("Número de vitórias", min_value=0, max_value=n, value=int(n * p_vitoria))

with col_b:
    probabilidade = binom.pmf(k, n, p_vitoria)
    st.write(f"Probabilidade de obter exatamente {k} vitórias em {n} jogos: **{probabilidade:.4f}**")

    x = np.arange(0, n + 1)
    y = binom.pmf(x, n, p_vitoria)

    fig_binom, ax_binom = plt.subplots(figsize=(5, 3))
    sns.barplot(x=x, y=y, palette="Blues", ax=ax_binom)
    ax_binom.set_title("Distribuição Binomial de Vitórias")
    ax_binom.set_xlabel("Número de Vitórias")
    ax_binom.set_ylabel("Probabilidade")
    st.pyplot(fig_binom)
