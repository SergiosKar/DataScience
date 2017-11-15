from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import DenseVector
from pyspark.sql import SparkSession

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

from pyspark.ml.regression import LinearRegression




spark = SparkSession.builder \
    .appName('Linear_Regression') \
    .master("local[*]") \
    .getOrCreate()

#spark_session.sparkContext.setLogLevel("WARN")

sc = spark.sparkContext

#load with csv
df = spark.read.csv("train.csv",header=True)
df.show()


'''data wrangling'''

#convert  to float
cols = [ 'Survived','Pclass', 'Age','SibSp','Parch','Fare']
for i in range(len(cols)):
    df = df.withColumn(cols[i],df[cols[i]].cast("float"))


#fillna
df= df.fillna({'Age': df.groupBy().mean('Age').first()[0] })
df= df.fillna({'Fare': df.groupBy().mean('Fare').first()[0] })


#group categories
df=df.withColumn('FamilySize' , df['SibSp'] + df['Parch'] + 1)

fare_udf=udf(lambda fare: 0.0 if (fare <= 7.91) else 1.0 if (fare>7.91 and fare<14.454)
            else 2.0 if (fare>14.454 and fare<31)
            else 3.0 ,FloatType())

df=df.withColumn("Fareband", fare_udf(df['Fare']))



#convert  categorical // or with udf
sex_udf=udf(lambda sex: 0.0 if (sex == 'male')  else 1.0 ,FloatType())
df=df.withColumn("Sex", sex_udf(df['Sex']))


#drop
df=df.drop('Parch','Fare','SibSp','Name','PassengerId','Ticket','Cabin','Embarked')
print(df.dtypes)

df.show()

'''Machine learning'''

# prepare labeled set

input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
lf = spark.createDataFrame(input_data, ["label", "features"])

lf.show(2)

#split
(trainingData, testData) = lf.randomSplit([0.8, 0.2])


# Linear Regression
lr = LinearRegression(labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the data to the model
lrModel = lr.fit(trainingData)
#dtModel= DecisionTreeClassifier(maxDepth=3, labelCol='label').fit(trainingData)
#rfModel = RandomForestClassifier(numTrees=100, labelCol='label').fit(trainingData)

# Generate predictions
predicted = lrModel.transform(testData)
# Extract the predictions and the "known" correct labels
predictions = predicted.select("prediction").rdd.map(lambda x: x[0])


labels = predicted.select("label").rdd.map(lambda x: x[0])
# Zip `predictions` and `labels` into a list
predictionAndLabel = predictions.zip(labels).collect()
print(predictionAndLabel[:5])



print('Linear Regression:',lrModel.summary.rootMeanSquaredError)
#print('Decision Tree:',testModel(dtModel))
#print('Random Forest:',testModel(rfModel))

spark.stop()




