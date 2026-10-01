FROM spark:4.1.3-python3
USER root
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && mkdir -p /opt/spark/conf \
 && echo "spark.jars.packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.3" >> /opt/spark/conf/spark-defaults.conf \
 && echo "spark.jars.ivy /lab/data/.ivy" >> /opt/spark/conf/spark-defaults.conf
USER spark
ENV PATH=/opt/spark/bin:$PATH
WORKDIR /lab
