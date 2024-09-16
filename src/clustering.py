from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

class KMeans:

    def __init__(self, k, features_col, logger) -> None:
        self.k = k
        self.log = logger
        self.featuers_col = features_col
        self.evaluator = ClusteringEvaluator(
            predictionCol='prediction',
            featuresCol=features_col,
            metricName='silhouette',
            distanceMeasure='squaredEuclidean'
        )

        self.kmeans = KMeans(featuresCol=features_col, k=self.k)

    def evaluate(self, predictions):
        return self.evaluator.evaluate(predictions)
        
    def predict(self, scaled_data):
        return self.kmeans.transform(scaled_data)
        