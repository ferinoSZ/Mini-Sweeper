import numpy as np
import pandas as pd

def extrair_features(tabuleiro, x, y): #Transformar uma célula do jogo em uma linha igual ao dataset. tambem recebe tabuleiro atual e coordenadas da célula.
    features = [] # aqui vai armazenar os valores das células vizinhas
    # percorre a janela 5x5 ao redor da célula
    for dx in range(-2, 3):
        for dy in range(-2, 3):

            # ignora a célula central
            if dx == 0 and dy == 0:
                continue

            nx = x + dx # Calculo da celula vizinha 
            ny = y + dy

            # se sair do tabuleiro → usa -2
            if (
                nx < 0
                or ny < 0
                or nx >= tabuleiro.shape[0]
                or ny >= tabuleiro.shape[1]
            ):
                features.append(-2)
            else:   
                features.append(tabuleiro[nx][ny])

    


    total_celulas = tabuleiro.size

    minas_suspeitas = np.sum(tabuleiro == -1)

    global_density = minas_suspeitas / total_celulas

    features.append(global_density)

    colunas = [
    "cell_-2_-2", "cell_-2_-1", "cell_-2_0", "cell_-2_1", "cell_-2_2",
    "cell_-1_-2", "cell_-1_-1", "cell_-1_0", "cell_-1_1", "cell_-1_2",
    "cell_0_-2", "cell_0_-1", "cell_0_1", "cell_0_2",
    "cell_1_-2", "cell_1_-1", "cell_1_0", "cell_1_1", "cell_1_2",
    "cell_2_-2", "cell_2_-1", "cell_2_0", "cell_2_1", "cell_2_2",
    "global_density"
]

    return pd.DataFrame([features], columns=colunas)


def prever_jogada_segura(tabuleiro_atual, modelo):


    tabuleiro = np.array(tabuleiro_atual)

    melhor_probabilidade = -1

    melhor_jogada = None

    for x in range(tabuleiro.shape[0]):
        for y in range(tabuleiro.shape[1]):

            if tabuleiro[x][y] == -1:

                features = extrair_features(tabuleiro, x, y)

                probabilidade_segura = modelo.predict_proba(features)[0][1]

                if probabilidade_segura > melhor_probabilidade:

                    melhor_probabilidade = probabilidade_segura

                    melhor_jogada = (x, y)

    if melhor_jogada is None:
        fechadas = list(zip(*np.where(tabuleiro == -1)))
        return fechadas[0] if fechadas else (0, 0)
    
    return melhor_jogada