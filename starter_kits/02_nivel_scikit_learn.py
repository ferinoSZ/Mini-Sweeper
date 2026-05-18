# ==============================================================================
# 🚀 STARTER KIT: SCIKIT-LEARN (Machine Learning Clássico)
# ==============================================================================
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def pipeline_scikit_learn(caminho_dados):

    # 1. ETL (Extração e Transformação)
    print("1. Carregando dados...")

    df = pd.read_csv(caminho_dados)

    # amostragem aleatória para evitar viés do dataset
    df = df.sample(n=1300000, random_state=42)

    # separação entre features e target
    X = df.drop("safe", axis=1)
    y = df["safe"]


    # 2. Divisão de Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # 3. Instanciação e Treinamento
    print("2. Treinando o modelo...")

    modelo = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_split=15,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1
        )

    modelo.fit(X_train, y_train)


    # análise de generalização
    train_acc = modelo.score(X_train, y_train)
    test_acc = modelo.score(X_test, y_test)

    print(f"Acurácia treino: {train_acc * 100:.2f}%")
    print(f"Acurácia teste: {test_acc * 100:.2f}%")


    # 4. Avaliação
    print("3. Avaliando o modelo...")

    previsoes = modelo.predict(X_test)

    acuracia = accuracy_score(y_test, previsoes)

    print(f"Acurácia final: {acuracia * 100:.2f}%")

    joblib.dump(modelo, "modelo_minesweeper.pkl")
    print("Modelo salvo com sucesso!! como 'modelo_minesweeper.pkl'")

    return modelo


# --- ÁREA DE TESTE ---
if __name__ == "__main__":

    modelo_pronto = pipeline_scikit_learn(
        "dataset/minesweeper_dataset/minesweeper_dataset.csv"
    )