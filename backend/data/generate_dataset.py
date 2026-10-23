"""
generate_dataset.py — Generador del dataset sintético de pares de materias.

Uso:
    python data/generate_dataset.py --seed 42 --n-materias 40 --out data/pares_materias.csv
"""
# TODO: implementar generación sintética según las reglas documentadas en el README
import argparse


def generate(seed: int, n_materias: int, out_path: str):
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-materias", type=int, default=40)
    parser.add_argument("--out", default="data/pares_materias.csv")
    args = parser.parse_args()
    generate(args.seed, args.n_materias, args.out)
