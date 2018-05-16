
  package org.apache.spark.ml.feature

  import org.apache.spark.ml.linalg.{Vector, SparseVector, DenseVector}
  import org.apache.spark.ml.Transformer
  import org.apache.spark.ml.util.Identifiable
  import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
  import org.apache.spark.ml.util._
  import org.apache.spark.ml.attribute._
  import org.apache.spark.ml.param._

  import org.apache.spark.sql.{Dataset, DataFrame}
  import org.apache.spark.ml.param._
  import org.apache.spark.sql.types.{StructField, StructType}
  // have sparse Vector
  import org.apache.spark.sql.functions.udf
  import sparkenv.SparkEnvironment._
  import spark.implicits._


  class Clip(override val uid: String = Identifiable.randomUID("clip")) extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {
    // mimic numpy.clip
    final val clip = new DoubleArrayParam(this, "clip", "Given an interval, passed as an array, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.")

    def setInputCol(value: String): this.type = set(inputCol, value)
    def setOutputCol(value: String): this.type = set(outputCol, value)
    def setClip(value: Array[Double]): this.type = set(clip, value)

    def getClip: Array[Double] = $(clip)

    setDefault(inputCol -> "features", outputCol -> "clipped_features", clip -> Array(-3d, 3d))

    def copy(extra: ParamMap): Clip = {
      defaultCopy(extra)
    }

    def transformSchema(schema: StructType): StructType = {
      val idx = schema.fieldIndex($(inputCol))
      val field = schema.fields(idx)
      schema.add( StructField($(outputCol), field.dataType, false))
    }

    def transform(df: Dataset[_]): DataFrame = {
      // val clip = udf{ in: org.apache.spark.ml.linalg.SparseVector =>
      val a = this.getClip.apply(0)
      val b = this.getClip.apply(1)
      val clippedFeature = udf{ in: org.apache.spark.ml.linalg.Vector =>
        in match {
          case x: DenseVector => new DenseVector( x.values.map{ e => if(e > b) b else if (e < a) a else e})
          case x: SparseVector => {
            val indices = x.indices
            val values = x.values.map{ e => if(e > b) b else if (e < a) a else e}
            new SparseVector( x.size, indices, values)
          }
        }
      }
      df.select(df.col("*"), clippedFeature(df.col($(inputCol))).as($(outputCol)))
    }
  }

  object Clip extends DefaultParamsReadable[Clip] {
    override def load(path: String): Clip = super.load(path)
  }
