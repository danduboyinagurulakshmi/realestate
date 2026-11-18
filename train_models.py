from pathlib import Path

from src.real_estate_pipeline import train_and_evaluate


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent / "india_housing_prices.csv"
    models_dir = Path(__file__).resolve().parent / "models"
    results = train_and_evaluate(data_path=data_path, output_dir=models_dir)
    print(results)
