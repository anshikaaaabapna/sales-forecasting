import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:

    def initiate_data_ingestion(self):

        df = pd.read_csv(
            "notebook/stores_sales_forecasting.csv",
            encoding="latin1"
        )

        os.makedirs("artifacts", exist_ok=True)

        raw_path = "artifacts/raw.csv"
        train_path = "artifacts/train.csv"
        test_path = "artifacts/test.csv"

        df.to_csv(raw_path, index=False)

        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        train_set.to_csv(train_path, index=False)
        test_set.to_csv(test_path, index=False)

        return train_path, test_path