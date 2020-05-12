import matplotlib
matplotlib.use("Agg")

from pyspark import SparkConf,SparkContext
from datetime import datetime
import math
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.functions import to_timestamp, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
import time
import numpy as np
import matplotlib.pyplot as plt

conf = SparkConf().setAppName("Assingment")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


#--------------- RDD AND MAPREDUCE QUERIES----------------------

def Haversine(f1,l1,f2,l2):
    a = math.sin((f2-f1)/2)**2+ math.cos(f1) * math.cos(f2) *math.sin((l2-l1)/2)**2
    c = 2 * math.atan2(math.sqrt(a),math.sqrt(1-a))
    d = 6371 * c
    return d

def Mapper_1(s):
    data = s.split(',')
    datetime_string_start = data[1]
    datetime_string_end = data[2]
    datetime_object_start = datetime.strptime(datetime_string_start, '%Y-%m-%d %H:%M:%S')
    datetime_object_end = datetime.strptime(datetime_string_end, '%Y-%m-%d %H:%M:%S')
    difference = datetime_object_end-datetime_object_start
    difference_in_minutes = difference.total_seconds()/60
    time = data[1].split(" ")[1].split(":")[0]
    return (time,difference_in_minutes)

def Filter_2(s):
    data = s.split(',')
    f1 = float(data[4])
    l1 = float(data[3])
    f2 = float(data[6])
    l2 = float(data[5])
    datetime_string_start = data[1]
    datetime_object_start = datetime.strptime(datetime_string_start, '%Y-%m-%d %H:%M:%S')
    longtitude = l1 >= -80.0 and l1<=-71.0 and l2 >= -80.0 and l2<=-71.0 # apofeugoume analithis times
    latitude =  f2>=40.0 and f2<=46.0 and f2>=40.0 and f2<=46.0 
    if (longtitude and latitude and datetime_object_start.day>=10):
        return True
    return False

def Mapper_2(s):
    data = s.split(',')
    f1 = float(data[4])
    l1 = float(data[3])
    f2 = float(data[6])
    l2 = float(data[5])
    distance = Haversine(f1,l1,f2,l2)
    datetime_string_start = data[1]
    datetime_string_end = data[2]
    datetime_object_start = datetime.strptime(datetime_string_start, '%Y-%m-%d %H:%M:%S')
    datetime_object_end = datetime.strptime(datetime_string_end, '%Y-%m-%d %H:%M:%S')
    difference = datetime_object_end-datetime_object_start
    if (difference.total_seconds()==0): return (data[0],-1)
    velocity = distance / difference.total_seconds()  #metra ana second
    return (data[0],velocity*3.6) #km ana wra

def Aggregate(arg,lista):
    for i in lista:
        if(arg[0] == i[0]):
            return True
    return False

#-------1)Query1---------------------
yellow_tripdata_1m = sc.textFile("hdfs://master:9000/yellow_tripdata_1m.csv")
start_time = time.time()
agreggated_1 = yellow_tripdata_1m.map(Mapper_1).aggregateByKey((0,0), lambda a,b: (a[0] + b, a[1] +1),lambda a,b: (a[0] + b[0], a[1] + b[1]))
query_1 = agreggated_1.mapValues(lambda v: v[0]/v[1]).sortBy(lambda a: a[0]).collect()
t1 = time.time() - start_time
print(query_1)

#----2)Query2---------------------
start_time = time.time()
agreggated_2 = yellow_tripdata_1m.filter(Filter_2).map(Mapper_2).top(5, key=lambda x: x[1])
yellow_tripvendors_1m = sc.textFile("hdfs://master:9000/yellow_tripvendors_1m.csv").map(lambda s: s.split(",")).map(lambda line: (line[0],line[1])).filter(lambda line: Aggregate(line,agreggated_2))
query_2 = sc.parallelize(agreggated_2).join(yellow_tripvendors_1m).collect()
t1 = time.time() - start_time
print(query_2)


#--------------- DATAFRAME AND SQL QUERIES----------------------
df_tripdata = sqlContext.read.csv("hdfs://master:9000/yellow_tripdata_1m.csv")
df_tripvendors = sqlContext.read.csv("hdfs://master:9000/yellow_tripvendors_1m.csv")

#-------1)Query1---------------------


#1a)Query1 with DataFrame API from csv
start_time = time.time()
df_tripdata.withColumn('duration',(to_timestamp(col('_c2')).cast(LongType()) - to_timestamp(col('_c1')).cast(LongType()))/60 )\
  .withColumn('start', substring(col("_c1"), 12, 2))\
  .groupBy("start").avg("duration").sort(asc("start")).show(25)   
t2 = time.time() - start_time

#1b)Query1 with DataFrame API from parquet
# start_time = time.time()
# df_tripdata.write.parquet("hdfs://master:9000/yellow_tripdata_1m.parquet")
# print("---yellow_tripdata_1m.parquet write in %s seconds ---" % (time.time() - start_time))
df_tripdata_parquet = sqlContext.read.parquet("hdfs://master:9000/yellow_tripdata_1m.parquet")

start_time = time.time()
df_tripdata_parquet.withColumn('duration',(to_timestamp(col('_c2')).cast(LongType()) - to_timestamp(col('_c1')).cast(LongType()))/60 )\
  .withColumn('start', substring(col("_c1"), 12, 2))\
  .groupBy("start").avg("duration").sort(asc("start")).show(25)
t3 = time.time() - start_time

#----2)Query2---------------------


#2a)Query2 with DataFrame API from csv
start_time = time.time()
df_tripdata.filter((substring(col("_c1"), 9, 2)>=10))\
    .withColumn("a", pow(sin((col("_c6")-col("_c4"))/2),2)+ cos(col("_c4")) * cos(col("_c6")) *pow(sin((col("_c5")-col("_c3"))/2),2) )\
    .withColumn("c",  2 * atan2(sqrt(col("a")),sqrt(1-col("a"))))\
    .withColumn("distance", col("c") * 6371)\
    .withColumn('duration',(to_timestamp(col('_c2')).cast(LongType()) - to_timestamp(col('_c1')).cast(LongType())) )\
    .withColumn("velocity", col("distance")/col("duration")*3.6)\
    .sort(col("velocity").desc())\
    .select("velocity","_c0").limit(5).join(df_tripvendors, df_tripvendors._c0 == df_tripdata._c0)\
    .withColumn("Vendor", col("_c1")).select("velocity", "Vendor").sort(col("velocity").desc())\
    .show()
t2 = time.time() - start_time
#2b)Query2 with DataFrame API from parquet
# start_time = time.time()
# df_tripvendors.write.parquet("hdfs://master:9000/yellow_tripvendors_1m.parquet")
# print("---yellow_tripvendors_1m.parquet write in %s seconds ---" % (time.time() - start_time))
df_tripvendors_parquet = sqlContext.read.parquet("hdfs://master:9000/yellow_tripvendors_1m.parquet")

start_time = time.time()
df_tripdata_parquet.filter((substring(col("_c1"), 9, 2)>=10))\
    .withColumn("a", pow(sin((col("_c6")-col("_c4"))/2),2)+ cos(col("_c4")) * cos(col("_c6")) *pow(sin((col("_c5")-col("_c3"))/2),2) )\
    .withColumn("c",  2 * atan2(sqrt(col("a")),sqrt(1-col("a"))))\
    .withColumn("distance", col("c") * 6371)\
    .withColumn('duration',(to_timestamp(col('_c2')).cast(LongType()) - to_timestamp(col('_c1')).cast(LongType())) )\
    .withColumn("velocity", col("distance")/col("duration")*3.6)\
    .sort(col("velocity").desc())\
    .select("velocity","_c0").limit(5).join(df_tripvendors_parquet, df_tripvendors_parquet._c0 == df_tripdata_parquet._c0)\
    .withColumn("Vendor", col("_c1")).select("velocity", "Vendor").sort(col("velocity").desc())\
    .show()
t3 = time.time() - start_time

objects = ("RDD", "SQL/csv", "SQL/parquet")
y_pos = np.arange(len(objects))
performance = [t1,t2,t3]
plt.barh(y_pos, performance, align="center", alpha=0.5)
plt.yticks(y_pos,objects)
plt.xlabel("Time(s)")
plt.title("Time needed for Query 2")
plt.show()
plt.savefig("Q2.png")