from pyspark.ml.feature import VectorAssembler, StandardScaler
from typing import List
from .logger import Logger
from .db_connector import MySQLConnector

class Preprocessor:
    def __init__(self, logger) -> None:
        self.features_name = 'features'
        self.scaled_features_name = 'scaled_features'
        self.separator = '\t'
        self.log = logger

    @Logger.cls_se_log("Загрузка датасета из БД")
    def load(self, db_connector: MySQLConnector, table: str, base_features: List[str]):
        dataset = db_connector.read(table)

        vector_assembler = VectorAssembler(
            inputCols=base_features,
            outputCol=self.features_name,
            handleInvalid='skip',
        )

        vectorized_data = vector_assembler.transform(dataset)
        return vectorized_data
    
    @Logger.cls_se_log("Применение к экземпларам датасета StandardScaler-трансформации")
    def apply_scale(self, vectorized_data):
        scaler = StandardScaler(
            inputCol=self.features_name,
            outputCol=self.scaled_features_name
        )

        scaler_model = scaler.fit(vectorized_data)
        scaled_data = scaler_model.transform(vectorized_data)

        return scaled_data