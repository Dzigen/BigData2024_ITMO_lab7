from pyspark.ml.feature import VectorAssembler, StandardScaler
from .logger import Logger

class Preprocessor:
    def __init__(self, logger) -> None:
        self.features_name = 'features'
        self.scaled_features_name = 'scaled_features'
        self.separator = '\t'
        self.log = logger

    @Logger.cls_se_log("Загрузка датасета в spark")
    def load(self, path_to_data, spark, base_features):
        dataset = spark.read.csv(
            path_to_data,
            header=True,
            inferSchema=True,
            sep=self.separator,
        )

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