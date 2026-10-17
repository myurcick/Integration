"""Train the model and persist the resulting artifact."""

from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .model import (
    CAT,
    NUM,
    TARGET,
    load_training_data,
    train_model,
)


ARTIFACTS = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
)


def main() -> None:
    """Train the model, report metrics, and persist the artifact."""
    
    df = load_training_data()

    # 1. Виділяємо ознаки (X) та цільову змінну (y)
    X = df[NUM + CAT]
    y = df[TARGET]

    # 2. Розбиваємо дані: 80% на навчання, 20% на тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Отримуємо пайплайн моделі
    model = train_model()
    
    # 3. ПЕРЕНАВЧАЄМО модель ТІЛЬКИ на тренувальній вибірці
    model.fit(X_train, y_train)

    # 4. Робимо прогноз ТІЛЬКИ на тестовій вибірці (нових даних для моделі)
    predictions = model.predict(X_test)

    # 5. Рахуємо метрики якості на тестовій вибірці
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Виводимо out-of-sample метрики
    print(
        f"[train] Ames out-of-sample: "
        f"R2={r2:.4f}, MAE={mae:.0f} USD"
    )

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_path = ARTIFACTS / "model.pkl"

    joblib.dump(
        model,
        artifact_path,
    )

    print(
        f"[train] artifact persisted: {artifact_path}"
    )

if __name__ == "__main__":
    main()
