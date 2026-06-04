import pandas as pd

from sklearn.model_selection import train_test_split

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_column = "Sales"

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    transformer = DataTransformation()

    preprocessor = transformer.get_preprocessor()

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    trainer = ModelTrainer()

    score = trainer.initiate_model_trainer(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print("R2 Score:", score)