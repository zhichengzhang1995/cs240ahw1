#!/bin/bash
################################################################################
#  A simple Scala based example for Spark
#  Designed to run on SDSC's Comet resource.
#  Mahidhar Tatineni, San Diego Supercomputer Center   June 2015
################################################################################
#SBATCH --job-name="sparkpython-demo"
#SBATCH --output="spark.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --export=ALL
#SBATCH -t 00:30:00

### Environment setup for Hadoop and Spark
module load spark
export PATH=/opt/hadoop/2.6.0/sbin:$PATH
export HADOOP_CONF_DIR=$HOME/mycluster.conf
export WORKDIR=`pwd`

myhadoop-configure.sh

### Start HDFS.  Starting YARN isn't necessary since Spark will be running in
### standalone mode on our cluster.
start-dfs.sh

### Load in the necessary Spark environment variables
source $HADOOP_CONF_DIR/spark/spark-env.sh

### Start the Spark masters and workers.  Do NOT use the start-all.sh provided
### by Spark, as they do not correctly honor $SPARK_CONF_DIR
myspark start

### Copy the data into HDFS
hdfs dfs -mkdir -p /user/$USER
#hdfs dfs -put $WORKDIR/facebook_combined.txt /user/$USER/
#hdfs dfs -put $WORKDIR/data/simple1 /user/$USER/simple1
hdfs dfs -put $WORKDIR/data/wiki /user/$USER/wiki

#run-example org.apache.spark.examples.graphx.LiveJournalPageRank facebook_combined.txt --numEPart=8
#spark-submit run_pagerank.py s /user/$USER/simple1 2 > simple12
#this works: pyspark run_pagerank.py s simple1 20 > temp
#it works:  spark-submit run_pagerank.py s simple1 20 > temp
spark-submit run_pagerank.py s /user/$USER/wiki 20 > wiki_out_20
#spark-submit run_pagerank.py s /user/$USER/wiki 20 > wiki20
#python utils/map_to_names.py temp data/wiki_mapping.json > wiki_out
### Shut down Spark and HDFS
myspark stop
stop-dfs.sh

### Clean up
myhadoop-cleanup.sh
