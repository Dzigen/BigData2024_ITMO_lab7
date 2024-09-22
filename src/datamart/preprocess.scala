
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame
import com.typesafe.scalalogging.Logger


object Preprocessor {

  private val logger = Logger("Logger")

  def load(df: DataFrame): DataFrame = {
    logger.info("Старт загрузки датасета из БД...")
    val outputCol = "features"
    val inputCols = "code" :: "created_t" :: "last_modified_t" :: "completeness"

    val vector_assembler = new VectorAssembler()
      .setInputCols(df.columns)
      .setOutputCol(outputCol)
      .setHandleInvalid("skip")
    val result = vector_assembler.transform(df)
    logger.info("Датасет загружен!")
    result
  }

  def apply_scale(df: DataFrame): DataFrame = {
    logger.info("Старт применения к экземпларам датасета StandardScaler-трансформации")
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaled_features")
    val scalerModel = scaler.fit(df)
    val result = scalerModel.transform(df)
    logger.info("Трансформация применена!")
    result
  }
}