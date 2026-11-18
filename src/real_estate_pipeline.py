from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parents[1] / "india_housing_prices.csv"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MLFLOW_DIR = Path(__file__).resolve().parents[1] / "mlruns"


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.copy()


def normalize_categorical(series: pd.Series) -> pd.Series:
    return series.fillna("Unknown").astype(str).str.strip()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    for col in [
        "BHK",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Year_Built",
        "Floor_No",
        "Total_Floors",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "Size_in_SqFt",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["Property_Type"] = normalize_categorical(frame.get("Property_Type", pd.Series(["Unknown"] * len(frame))))
    frame["Furnished_Status"] = normalize_categorical(frame.get("Furnished_Status", pd.Series(["Unfurnished"] * len(frame))))
    frame["Public_Transport_Accessibility"] = normalize_categorical(frame.get("Public_Transport_Accessibility", pd.Series(["Low"] * len(frame))))
    frame["Parking_Space"] = normalize_categorical(frame.get("Parking_Space", pd.Series(["No"] * len(frame))))
    frame["Security"] = normalize_categorical(frame.get("Security", pd.Series(["No"] * len(frame))))
    frame["Amenities"] = frame.get("Amenities", pd.Series([""] * len(frame))).fillna("")
    frame["Facing"] = normalize_categorical(frame.get("Facing", pd.Series(["North"] * len(frame))))
    frame["Owner_Type"] = normalize_categorical(frame.get("Owner_Type", pd.Series(["Owner"] * len(frame))))
    frame["Availability_Status"] = normalize_categorical(frame.get("Availability_Status", pd.Series(["Ready_to_Move"] * len(frame))))
    frame["State"] = normalize_categorical(frame.get("State", pd.Series(["Unknown"] * len(frame))))
    frame["City"] = normalize_categorical(frame.get("City", pd.Series(["Unknown"] * len(frame))))
    frame["Locality"] = normalize_categorical(frame.get("Locality", pd.Series(["Unknown"] * len(frame))))

    frame["Size_in_SqFt"] = frame["Size_in_SqFt"].fillna(frame["Size_in_SqFt"].median())
    frame["BHK"] = frame["BHK"].fillna(frame["BHK"].median())
    frame["Price_in_Lakhs"] = frame["Price_in_Lakhs"].fillna(frame["Price_in_Lakhs"].median())
    frame["Price_per_SqFt"] = frame["Price_per_SqFt"].fillna(frame["Price_in_Lakhs"] * 100000 / frame["Size_in_SqFt"])
    frame["Year_Built"] = frame["Year_Built"].fillna(frame["Year_Built"].median())
    frame["Age_of_Property"] = frame["Age_of_Property"].fillna(np.maximum(0, 2025 - frame["Year_Built"]))
    frame["Nearby_Schools"] = frame["Nearby_Schools"].fillna(frame["Nearby_Schools"].median())
    frame["Nearby_Hospitals"] = frame["Nearby_Hospitals"].fillna(frame["Nearby_Hospitals"].median())
    frame["Floor_No"] = frame["Floor_No"].fillna(frame["Floor_No"].median())
    frame["Total_Floors"] = frame["Total_Floors"].fillna(frame["Total_Floors"].median())

    frame["Amenities_Count"] = frame["Amenities"].fillna("").str.split(",").apply(lambda values: len([v for v in values if v and v.strip()]))
    frame["Parking_Indicator"] = frame["Parking_Space"].str.lower().eq("yes").astype(int)
    frame["Security_Indicator"] = frame["Security"].str.lower().isin(["yes", "gated", "guard", "cctv"]).astype(int)
    frame["Transport_Score"] = frame["Public_Transport_Accessibility"].str.lower().map({"low": 1, "medium": 2, "high": 3}).fillna(2).astype(int)
    frame["Furnished_Score"] = frame["Furnished_Status"].str.lower().map({"unfurnished": 0, "semi-furnished": 1, "semi": 1, "furnished": 2}).fillna(1).astype(int)
    frame["Value_to_Size"] = frame["Price_in_Lakhs"] / frame["Size_in_SqFt"].replace(0, np.nan)
    frame["Value_to_Size"] = frame["Value_to_Size"].fillna(frame["Value_to_Size"].median())

    frame["Projected_Annual_Appreciation"] = (
        0.06
        + 0.012 * frame["Transport_Score"]
        + 0.010 * frame["Nearby_Schools"] / 10
        + 0.009 * frame["Nearby_Hospitals"] / 10
        + 0.008 * frame["Parking_Indicator"]
        + 0.010 * frame["Amenities_Count"] / 5
        + 0.006 * frame["Furnished_Score"]
    ).clip(lower=0.04, upper=0.18)

    frame["Five_Year_Projected_Price"] = frame["Price_in_Lakhs"] * (1 + frame["Projected_Annual_Appreciation"] * 5)
    frame["Good_Investment"] = (frame["Projected_Annual_Appreciation"] > 0.11).astype(int)

    numeric_cols = [
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Year_Built",
        "Floor_No",
        "Total_Floors",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "Amenities_Count",
        "Parking_Indicator",
        "Security_Indicator",
        "Transport_Score",
        "Furnished_Score",
        "Value_to_Size",
        "Projected_Annual_Appreciation",
    ]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].isna().any():
            frame[col] = frame[col].fillna(frame[col].median())

    return frame


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "ID",
        "Five_Year_Projected_Price",
        "Good_Investment",
        "Projected_Annual_Appreciation",
        "Amenities",
        "Parking_Space",
        "Security",
    }
    return [col for col in df.columns if col not in excluded]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_model_pipelines(X: pd.DataFrame, y_class: pd.Series, y_reg: pd.Series) -> Tuple[Pipeline, Pipeline]:
    preprocessor = build_preprocessor(X)

    classification_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced", n_jobs=-1)),
        ]
    )

    regression_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)),
        ]
    )

    classification_pipeline.fit(X, y_class)
    regression_pipeline.fit(X, y_reg)
    return classification_pipeline, regression_pipeline


def train_and_evaluate(data_path: str | Path = DATA_PATH, output_dir: str | Path = MODELS_DIR) -> Dict[str, Any]:
    df = engineer_features(load_data(data_path))
    sample_size = min(len(df), 40000)
    df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    feature_columns = get_feature_columns(df)
    X = df[feature_columns]
    y_class = df["Good_Investment"]
    y_reg = df["Five_Year_Projected_Price"]

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X,
        y_class,
        y_reg,
        test_size=0.2,
        random_state=42,
        stratify=y_class,
    )

    classifier_pipeline, regressor_pipeline = build_model_pipelines(X_train, y_class_train, y_reg_train)
    class_pred = classifier_pipeline.predict(X_test)
    reg_pred = regressor_pipeline.predict(X_test)

    accuracy = accuracy_score(y_class_test, class_pred)
    mae = mean_absolute_error(y_reg_test, reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, reg_pred))
    r2 = r2_score(y_reg_test, reg_pred)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = output_dir / "investment_classifier.joblib"
    regressor_path = output_dir / "five_year_value_regressor.joblib"
    joblib.dump(classifier_pipeline, classifier_path)
    joblib.dump(regressor_pipeline, regressor_path)

    trusted_types = ["numpy.dtype", "sklearn.tree._tree.Tree"]

    with mlflow.start_run(run_name="real-estate-investment-advisor") as active_run:
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_metric("classification_accuracy", accuracy)
        mlflow.log_metric("regression_mae", mae)
        mlflow.log_metric("regression_rmse", rmse)
        mlflow.log_metric("regression_r2", r2)
        mlflow.sklearn.log_model(classifier_pipeline, artifact_path="classifier_model", skops_trusted_types=trusted_types)
        mlflow.sklearn.log_model(regressor_pipeline, artifact_path="regressor_model", skops_trusted_types=trusted_types)
        mlflow.set_tag("project", "real_estate_investment_advisor")

    result = {
        "classification_accuracy": float(accuracy),
        "regression_mae": float(mae),
        "regression_rmse": float(rmse),
        "regression_r2": float(r2),
        "classifier_path": str(classifier_path),
        "regressor_path": str(regressor_path),
        "feature_columns": feature_columns,
    }

    metadata_path = output_dir / "training_summary.json"
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_model_bundle(model_dir: str | Path = MODELS_DIR) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    classifier = joblib.load(model_dir / "investment_classifier.joblib")
    regressor = joblib.load(model_dir / "five_year_value_regressor.joblib")
    return {"classifier": classifier, "regressor": regressor}


def prepare_single_property(raw_input: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([raw_input])
    prepared = engineer_features(df)
    feature_columns = get_feature_columns(prepared)
    return prepared[feature_columns]


def predict_property(raw_input: Dict[str, Any], model_bundle: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if model_bundle is None:
        model_bundle = load_model_bundle()

    prepared = prepare_single_property(raw_input)
    classifier = model_bundle["classifier"]
    regressor = model_bundle["regressor"]

    investment_probability = classifier.predict_proba(prepared)[0, 1]
    investment_flag = int(investment_probability >= 0.5)

    projected_price = float(regressor.predict(prepared)[0])
    appreciation = projected_price / float(raw_input.get("Price_in_Lakhs", 1.0)) - 1.0

    return {
        "good_investment": bool(investment_flag),
        "confidence": float(investment_probability),
        "projected_5_year_price_lakhs": round(projected_price, 2),
        "expected_appreciation": round(appreciation * 100, 2),
    }


def save_recommendation_summary(data_path: str | Path = DATA_PATH, output_path: str | Path = "plots/eda_summary.json") -> Dict[str, Any]:
    df = engineer_features(load_data(data_path))

    summary = {
        "price_summary": {
            "mean": round(float(df["Price_in_Lakhs"].mean()), 2),
            "median": round(float(df["Price_in_Lakhs"].median()), 2),
            "max": round(float(df["Price_in_Lakhs"].max()), 2),
            "min": round(float(df["Price_in_Lakhs"].min()), 2),
        },
        "top_cities_by_average_price": df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(10).round(2).to_dict(),
        "good_investment_share": round(float(df["Good_Investment"].mean() * 100), 2),
        "price_by_property_type": df.groupby("Property_Type")["Price_per_SqFt"].mean().sort_values(ascending=False).round(2).to_dict(),
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = train_and_evaluate()
    print(json.dumps(result, indent=2))
