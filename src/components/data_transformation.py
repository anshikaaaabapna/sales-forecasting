import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class DataTransformation:

    def get_preprocessor(self):

        numerical_columns = [
            "Quantity",
            "Discount",
            "Profit"
        ]

        categorical_columns = [
            "Category",
            "Sub-Category",
            "Region",
            "Segment",
            "Ship Mode"
        ]

        num_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns)
            ]
        )

        return preprocessor