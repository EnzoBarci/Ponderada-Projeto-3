import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importa os dados
df = pd.read_csv('medical_examination.csv')

# Adiciona a coluna 'overweight' (acima do peso)
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# Normaliza os dados para que 0 sempre represente um valor bom e 1 represente um valor ruim.
# Se o valor de 'cholesterol' ou 'gluc' for 1, faz o valor 0. Se o valor for maior que 1, faz o valor 1.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Desenha o Gráfico Categórico
def draw_cat_plot():
    # Cria um DataFrame para o gráfico categórico usando `pd.melt` com os valores de 'cholesterol', 'gluc', 'smoke', 'alco', 'active' e 'overweight'.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # Agrupa e reformata os dados para dividi-los por 'cardio'. Mostra as contagens de cada característica.
    # É necessário renomear uma das colunas para que o catplot funcione corretamente.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    df_cat = df_cat.rename(columns={'size': 'total'})
    print(df_cat.head())

    # Desenha o gráfico categórico com 'sns.catplot()'
    g = sns.catplot(data=df_cat, kind="bar", x="variable", y="total", hue="value", col="cardio")

    # Obtém a figura para o output
    fig = g.fig

    # Não modifique as próximas duas linhas
    fig.savefig('catplot.png')
    return fig

# Desenha o Mapa de Calor
def draw_heat_map():
    # Limpa os dados
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # Calcula a matriz de correlação
    corr = df_heat.corr()

    # Gera uma máscara para o triângulo superior da matriz
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Configura a figura do matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))

    # Desenha o mapa de calor com 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, cmap='rocket', fmt=".1f", linewidths=.5, vmin=-.08, vmax=.24, ax=ax)

    fig.savefig('heatmap.png')
    return fig
