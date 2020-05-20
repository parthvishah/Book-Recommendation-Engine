#!/usr/bin/env python

import pandas as pd
import sys
import math
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.functions import col, size,  count
from pyspark.sql.types import IntegerType
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import time
import pickle
import re
import seaborn as sns
import pickle
import os
import time
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

conf = SparkConf()
conf.set("spark.executor.memory", "6G")
conf.set("spark.driver.memory", "15G")
conf.set("spark.executor.cores", "4")

conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.default.parallelism", "4")
conf.setMaster('local[4]')

spark = SparkSession \
    .builder.config(conf=conf) \
    .appName("spark-santander-recommendation").getOrCreate()


def main(spark, data_file):
    '''Main routine for supervised evaluation

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the parquet file to load
    '''


    # Load the dataframe from data file
    #input_data = spark.read.parquet(data_file)
    df = spark.read.csv(data_file, header = True)

    #df = df.limit(500000)
    df = df.filter(df.user_id.isNotNull())
    print('1')
    df.show()
    

    df_unique_user = df.select("user_id").distinct()
    df_unique_user = df_unique_user.selectExpr("user_id as uid")

    percent = df_unique_user.count()
    percent = 0.001*percent
    percent = math.floor(percent)
    print(percent)

    df_unique_user = df_unique_user.limit(percent)
    print('df unique user')
    df_unique_user.show()
    df_final = df.join(df_unique_user, df.user_id == df_unique_user.uid, "inner").select(df.user_id, df.book_id, df.is_read, df.rating, df.is_reviewed)
    df = df_final
    print('2')
    df.show()
    #FILTER USERS < 10
    counts=df.groupBy('user_id').count().selectExpr("user_id as uid", "count as count")
    df = df.join(counts, df.user_id == counts.uid).filter(F.col("count") > 10).drop('uid','count')
    print('3')
    df.show()


    from pyspark.sql.types import DoubleType

    df=df.filter(df.rating.isNotNull())
    df=df.withColumn("rating", df["rating"].cast(DoubleType()))
    from pyspark.ml.feature import StringIndexer

    stage_1 = StringIndexer(inputCol='user_id', outputCol='user_id_index')
    #df = stage_1.setHandleInvalid("keep").fit(df).transform(df)
    df = stage_1.fit(df).transform(df)

    stage_2 = StringIndexer(inputCol='book_id', outputCol='book_id_index')
    #transformed = stage_2.setHandleInvalid("keep").fit(df).transform(df)
    df = stage_2.fit(df).transform(df)


    user_id = df.select("user_id").distinct()
    uid = df.select(F.collect_set('user_id').alias('user_id')).first()['user_id']


    #get the count of each user_id
    counts = df.groupBy('user_id').count() #Show count of each user_id
    counts = counts.selectExpr('user_id as user_id','count as n') #Rename count as n

    #Create Train Test and Validation sets 60-20-20
    train_size = int(0.6 * len(uid))
    vali_size = train_size + int(0.2 * len(uid))
    test_size = vali_size + int(0.2 * len(uid))

    train_set = uid[:train_size]
    vali_set = uid[train_size:vali_size]
    test_set = uid[vali_size:]

    train_set=df.filter(df.user_id.isin(train_set))
    vali_set=df.filter(df.user_id.isin(vali_set))
    test_set=df.filter(df.user_id.isin(test_set))
#-----------------------------------------------


    # In[8]:


    vali_uid = vali_set.select(F.collect_set('user_id').alias('user_id')).first()['user_id']
    test_uid = test_set.select(F.collect_set('user_id').alias('user_id')).first()['user_id']

    #For each validation user, use half of their interactions for training,
    validict={i : 0.5 for i in vali_uid}
    new_vali = vali_set.sampleBy("user_id", fractions=validict, seed=40)

    testdict={i : 0.5 for i in test_uid}
    new_test = test_set.sampleBy("user_id", fractions=testdict, seed=40)

    vali_set = vali_set.exceptAll(new_vali)
    train_set = train_set.union(new_vali)

    test_set = test_set.exceptAll(new_test)
    train_set = train_set.union(new_test)
    train_set.show()


# # ALS

# In[40]:


    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.recommendation import ALS
    from pyspark.mllib.evaluation import RankingMetrics
    from pyspark.sql.functions import expr
    import itertools as it
    
    rank_ = [5,10,15]
    regParam_ = [ 0.01, 0.1,1.0 ]
    alpha_ = [ 1,2, 5]
    param_grid = it.product(rank_, regParam_, alpha_)
    vals_list= []
    stats =[]
    rmse_list = []
    best_map = 999999999
    best_model = None
    for i in param_grid:
        print('Start Training for {}'.format(i))
        als = ALS(rank = i[0], maxIter=10, regParam=i[1],alpha = i[2], userCol="user_id_index", itemCol="book_id_index", ratingCol='rating',
        nonnegative=True, coldStartStrategy="drop")
        model = als.fit(train_set)
        user_subset = vali_set.select("user_id_index").distinct()
        userRecs = model.recommendForUserSubset(user_subset, 500)
        from pyspark.sql.functions import expr
        print('Recommended')
        true_label = test_set.select('user_id_index', 'book_id_index')\
                .groupBy('user_id_index')\
                .agg(expr('collect_list(book_id_index) as true_item'))
        pred_label = userRecs.select('user_id_index','recommendations.book_id_index')
        print('pred_label')
        pred_true_rdd = pred_label.join(F.broadcast(true_label), 'user_id_index', 'inner') \
                .rdd \
                .map(lambda row: (row[1], row[2]))
        print('pred_true_rdd')
        metrics = RankingMetrics(pred_true_rdd)
        map_ = metrics.meanAveragePrecision
        ndcg = metrics.ndcgAt(500)
        mpa = metrics.precisionAt(500)
        evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
        predictions=model.transform(vali_set)
        predsDF = predictions.filter(predictions.rating.between(3,5)).collect()
        predsDF = spark.createDataFrame(predsDF)

        rmse = evaluator.evaluate(predsDF)
        rmse_list.append(rmse)
        if map_ < best_map:
                        best_model = model
                        best_map = map_
                        #print('New best model')
                        stats.append([i[0], i[1], i[2], rmse])

        columns =['Alpha','Rank','RegParam','MAP', 'Precision','NDGC','RMSE']
        vals_list.append((i[2],i[0],i[1],map_,mpa,ndcg,rmse))
        print('MAP: %f' %map_)
        print('Precision: %f' %mpa)
        print('NDCG: %f' %ndcg)
        print('rmse %f:' %rmse)
        plt.scatter(i[0],rmse)
        #plt.pause(0.05)
    plt.show()




    #als=ALS(maxIter=5,regParam=0.09,rank=200,userCol="user_id_index",itemCol="book_id_index",ratingCol="rating",coldStartStrategy="drop",nonnegative=True)
    #model=als.fit(train_set)

    #evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
    #predictions=model.transform(vali_set)

    predictions.show()
    convertToFloat = lambda lines: [double(x) for x in vals_list]
    #from pyspark.sql.types import *

    mySchema = StructType([ StructField("Alpha", IntegerType(), True)\

                           ,StructField("Rank", IntegerType(), True)\

                           ,StructField("Reg_Param", DoubleType())\

                           ,StructField("MAP ", DoubleType(), True)\

                           ,StructField("Precision", DoubleType(), True)\

                           ,StructField("NDGC", DoubleType(), True)\

                           ,StructField("RMSE", DoubleType(), True)])
    df = spark.createDataFrame(vals_list,schema=mySchema)
    df.show()
    
    
        #Evaluation of test set
    #print('Finish Training for {}'.format(i))
    user_subset = test_set.select("user_id_index").distinct()
    userRecs = best_model.recommendForUserSubset(user_subset, 500)
    
    true_label = test_set.select('user_id_index', 'book_id_index')\
                .groupBy('user_id_index')\
                .agg(expr('collect_list(book_id_index) as true_item'))
    pred_label = userRecs.select('user_id_index','recommendations.book_id_index')

    pred_true_rdd = pred_label.join(F.broadcast(true_label), 'user_id_index', 'inner') \
                .rdd \
                .map(lambda row: (row[1], row[2]))
    metrics = RankingMetrics(pred_true_rdd)
    map_ = metrics.meanAveragePrecision
    ndcg = metrics.ndcgAt(500)
    mpa = metrics.precisionAt(500)
    evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
    predictions=model.transform(test_set)
    predsDF = predictions.filter(predictions.rating.between(3,5)).collect()
    predsDF = spark.createDataFrame(predsDF)

    rmse = evaluator.evaluate(predsDF)
    print('Test Metrics:')
    print('MAP: %f' %map_)
    print('Precision: %f' %mpa)
    print('NDCG: %f' %ndcg)
    print('rmse %f:' %rmse)
    
    #Latent Factors
    ufac_df = best_model.userFactors.toPandas()
    ifac_df = best_model.itemFactors.toPandas()
    ufac_matrix = np.vstack(ufac_df.features.values)
    ifac_matrix = np.vstack(ifac_df.features.values)

    import seaborn as sns
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    palette = sns.color_palette("hls", 10)
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.manifold import TSNE

    X = ufac_matrix
    Y = ifac_matrix
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    Y_embedded = tsne.fit_transform(Y)
    plot_users = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], legend='full', palette=palette)
    plot_items = sns.scatterplot(Y_embedded[:,0], Y_embedded[:,1], legend='full')
    
    pkl.dump(ufac_matrix, open('ufac_matrix.pkl','wb'))
    pkl.dump(ifac_matrix, open('ifac_matrix.pkl','wb'))



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('als_test').getOrCreate()
    #memory = f"{math.floor(psutil.virtual_memory()[1]*.9) >> 30}g" #for local but for cluster did a fixed number
    '''spark = (SparkSession.builder
             .appName('als')
             .master('yarn')
             .config('spark.executor.memory', memory)
             .config('spark.driver.memory', memory)
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")'''

    # And the location to store the trained model

    # Get the filename from the command line
    data_file = sys.argv[1]

    # Call our main routine
    main(spark, data_file)
