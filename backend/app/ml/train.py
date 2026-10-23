"""train.py — Entrenamiento y comparación de modelos."""
# TODO: implementar comparación de modelos (regresión lineal, árbol, KNN)
# y guardado del mejor modelo con joblib


def train(data_path: str, out_path: str):
    """Entrena los modelos, imprime métricas y guarda el mejor en out_path."""
    raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train(args.data, args.out)
