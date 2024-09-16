from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

class KMeansConnector:

    def __init__(self, k, features_col, logger) -> None:
        self.k = k
        self.log = logger
        self.features_col = features_col
        self.evaluator = ClusteringEvaluator(
            predictionCol='prediction',
            featuresCol=features_col,
            metricName='silhouette',
            distanceMeasure='squaredEuclidean'
        )

        self.kmeans = KMeans(featuresCol=self.features_col, k=self.k)

    def fit(self, data):
        self.model = self.kmeans.fit(data)

    def evaluate(self, predictions):
        return self.evaluator.evaluate(predictions)
        
    def predict(self, data):
        return self.model.transform(data)
        