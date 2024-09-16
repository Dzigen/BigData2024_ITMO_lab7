from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from .logger import Logger

class KMeansConnector:

    def __init__(self, k, features_col, logger, metric_name: str = 'silhouette') -> None:
        self.k = k
        self.log = logger
        self.eval_metric = metric_name
        self.features_col = features_col
        self.evaluator = ClusteringEvaluator(
            predictionCol='prediction',
            featuresCol=features_col,
            metricName=self.eval_metric,
            distanceMeasure='squaredEuclidean'
        )

        self.kmeans = KMeans(featuresCol=self.features_col, k=self.k)

    @Logger.cls_se_log("Обучение модели кластеризации")
    def fit(self, data):
        self.model = self.kmeans.fit(data)

    @Logger.cls_se_log("Оценка качества модели кластеризации")
    def evaluate(self, predictions):
        return self.evaluator.evaluate(predictions)
    
    @Logger.cls_se_log("Получение предсказаний от обученной модели")
    def predict(self, data):
        return self.model.transform(data)
        