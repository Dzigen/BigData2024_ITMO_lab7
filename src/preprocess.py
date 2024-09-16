from pyspark.ml.feature import VectorAssembler, StandardScaler

class Preprocessor:
    def __init__(self, logger) -> None:
        self.features_colname = 'features'
        self.log = logger

    def load(self, path_to_data, spark, base_features):
        dataset = spark.read.csv(
            path_to_data,
            header=True,
            inferSchema=True,
            sep='\t',
        )

        vector_assembler = VectorAssembler(
            inputCols=base_features,
            outputCol=self.features_colname,
            handleInvalid='skip',
        )

        vectorized_data = vector_assembler.transform(dataset)
        return vectorized_data
    
    def perform_scale(self, vectorized_data):
        scaler = StandardScaler(
            inputCol=self.features_colname,
            outputCol=self.features_colname
        )

        scaler_model = scaler.fit(vectorized_data)
        scaled_data = scaler_model.transform(vectorized_data)

        return scaled_data.collect()