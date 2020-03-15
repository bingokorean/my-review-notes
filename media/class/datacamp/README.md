# PySpark Courses from DataCamp


### Contents

* [Introduction to PySpark](#Introduction-to-PySpark)
   1. Getting to know PySpark
   2. Manipulating data
   3. Getting started with machine learning pipelines
   4. Model tuning and selection
* [Big Data Fundamentals with PySpark](#Big-Data-Fundamentals-with-PySpark)
   1. ...
   2. ...
   3. PySpark SQL & DataFrames
   4. ...
* [Clearning Data with PySpark](#Clearning-Data-with-PySpark)
   1. DataFrame details
   2. Manipulating DataFrames in the real world
   3. Improving Performance
   4. Complex processing and data pipelines
* [Building Recommendation Engines with PySpark](#Building-Recommendation-Engines-with-PySpark)
   1. Recommendations Are Everywhere
   2. How does ALS work?



<br>




## Introduction to PySpark

### 1. Getting to know PySpark

Pyspark에서 Spark cluster와 연결하는 법은 해당 클래스 객체를 생성하는 일은 'SparkContext 클래스' 에서 담당한다. <br>
클러스터의 속성은  SparkContext 객체 생성시 인자를 통해 명시한다.

클러스터 속성들은 'SparkConf 클래스' 를 통해 생성될 수 있다.

간단한 계산은 오히려 spark에서 오래 걸릴 수도 있다. <br>
spark는 대용량 데이터에 맞는 복잡한 계산으로 최적화되어 있기 때문이다.

```python
# SparkContext 를 sc 에 생성했다고 가정.

# Verify SparkContext
print(sc)

# Print Spark version
print(sc.version)
```
```
<SparkContext master=local[*] appName=pyspark-shell>
2.3.1
```

Spark의 코어 자료구조는 Resilient Distributed Dataset(RDD)이다. <br>
이는 spark가 클러스터의 각 노드에 맞게 데이터를 분할할 수 있도록 해준다. <br>
RDD는 low-level object이므로 작업하기 어렵다. 대신에 Spark DataFrame abstraction을 사용한다.

Spark DataFrame은 SQL 테이블과 비슷하게 동작하기에 이해하기 쉽다. <br>
RDD보다 복잡한 계산에 더 최적화가 되어있다.

RDD를 사용할 때는 데이터 사이언티스트가 어떻게 쿼리를 최적화 하느냐에 따라 성능이 좌우된다. <br>
하지만, DataFrame은 이러한 최적화가 자동적으로 내장되어 있다.

Spark DataFrame을 사용하기 위해서 SparkContext 로부터 SparkSession 을 생성해야 한다. <br>
SparkContext를 클러스터와 연결해주는 매개체, SparkSession은 연결 매개체와 상호 작용하는 인터페이스라 생각할 수 있다.

SparkSession을 여러 개 생성하면 SparkContext에서는 이슈가 발생한다. <br>
이를 방지하기 위해 getOrCreat() 함수를 사용한다. 이미 존재하면 새로 만들지 않고 기존의 것을 리턴한다.

```python
# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession

# Create my_spark
my_spark = SparkSession.builder.getOrCreate()

# Print my_spark
print(my_spark)
```
```
<pyspark.sql.session.SparkSession object at 0x7f0c0ffaa128>
```

SparkSession 을 만들면, 클러스터에 있는 데이터를 확인할 수 있다. <br>
SparkSession은 catalog 속성을 가진다. 이는 cluster 안에 있는 모든 데이터의 리스트를 가진다. <br>
catalog.listTables() 함수로 cluster에 있는 모든 테이블의 이름을 확인할 수 있다.

```python
# Print the tables in the catalog
print(spark.catalog.listTables())
>>>
[Table(name='flights', database=None, description=None, tableType='TEMPORARY', isTemporary=True)]
```

DataFrame의 장점 중 하나는 SQL 쿼리를 Spark cluster에 날릴 수 있다는 점이다.

```python
query = "FROM flights SELECT * LIMIT 10"

# Get the first 10 rows of flights
flights10 = spark.sql(query)

# Show the results
flights10.show()
```
```
>>>
+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+----+--------+--------+----+------+
|year|month|day|dep_time|dep_delay|arr_time|arr_delay|carrier|tailnum|flight|origin|dest|air_time|distance|hour|minute|
+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+----+--------+--------+----+------+
|2014|   12|  8|     658|       -7|     935|       -5|     VX| N846VA|  1780|   SEA| LAX|     132|     954|   6|    58|
|2014|    1| 22|    1040|        5|    1505|        5|     AS| N559AS|   851|   SEA| HNL|     360|    2677|  10|    40|
|2014|    3|  9|    1443|       -2|    1652|        2|     VX| N847VA|   755|   SEA| SFO|     111|     679|  14|    43|
|2014|    4|  9|    1705|       45|    1839|       34|     WN| N360SW|   344|   PDX| SJC|      83|     569|  17|     5|
|2014|    3|  9|     754|       -1|    1015|        1|     AS| N612AS|   522|   SEA| BUR|     127|     937|   7|    54|
|2014|    1| 15|    1037|        7|    1352|        2|     WN| N646SW|    48|   PDX| DEN|     121|     991|  10|    37|
|2014|    7|  2|     847|       42|    1041|       51|     WN| N422WN|  1520|   PDX| OAK|      90|     543|   8|    47|
|2014|    5| 12|    1655|       -5|    1842|      -18|     VX| N361VA|   755|   SEA| SFO|      98|     679|  16|    55|
|2014|    4| 19|    1236|       -4|    1508|       -7|     AS| N309AS|   490|   SEA| SAN|     135|    1050|  12|    36|
|2014|   11| 19|    1812|       -3|    2352|       -4|     AS| N564AS|    26|   SEA| ORD|     198|    1721|  18|    12|
+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+----+--------+--------+----+------+
```

가끔, 대용량 데이터를 Spark 쿼리로 처리하고 집계된 결과를 more manageable한 Pandas DataFrame으로 표현하고 싶을 때가 있다.

```python
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"

# Run the query
flight_counts = spark.sql(query)

# Convert the results to a pandas DataFrame
pd_counts = flight_counts.toPandas()

# Print the head of pd_counts
print(pd_counts.head())
```
```
>>>
  origin dest    N
0    SEA  RNO    8
1    SEA  DTW   98
2    SEA  CLE    2
3    SEA  LAX  450
4    PDX  SEA  144
```

Pandas DataFrame를 Spark cluster에 저장하고 싶을 때가 있을 것이다. createDataFrame 함수는 pandas DataFrame을 받고 Spark DataFrame을 리턴해준다. 이 함수의 출력은 locally하게 저장될 뿐, SparkSession catalog에는 저장되지 않는다. 이 의미는 모든 Spark DataFrame method를 사용할 수는 있지만, data 자체의 접근을 할 수 없다는 뜻이다. 예를 들어, pandas DataFrame을 참조하는 SQL 쿼리를 날리면 에러가 발생한다. Pandas DataFrame에 접근하기 위해서 temporary table에 임시로 저장할 필요가 있다.

Spark DataFrame method인 .createTempView()를 사용하면 된다. 해당 DataFrame을 catalog의 table에 등록해준다. 하지만 이는 일시적이며, 지금 사용하는 특정 SparkSession에서만 접근할 수 있다. (.createOrReplaceTempView() 함수를 쓰자. 이는 없으면 만들고, 있으면 업데이트함으로써 duplicate table을 방지해준다)

```python
# Create pd_temp
pd_temp = pd.DataFrame(np.random.random(10))

# Create spark_temp from pd_temp
spark_temp = spark.createDataFrame(pd_temp)

# Examine the tables in the catalog
print(spark.catalog.listTables())
>>>
[Table(name='flights', database=None, description=None, tableType='TEMPORARY', isTemporary=True)]

# Add spark_temp to the catalog
spark_temp.createOrReplaceTempView("temp")

# Examine the tables in the catalog again
print(spark.catalog.listTables())
>>>
[Table(name='flights', database=None, description=None, tableType='TEMPORARY', isTemporary=True), Table(name='temp', database=None, description=None, tableType='TEMPORARY', isTemporary=True)]
```

다음 다이어그램을 통해 Spark DataFrame이 다른 것들과 어떻게 상호 작용하는지 확인할 수 있습니다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/media/class/datacamp/images/intro_to_pyspark/pic_1.PNG" width="70%" height="70%"></p>

SparkSession 에는 .read라는 함수로 다양한 데이터 소스를 Spark DataFrame으로 불러올 수 있다. 한 가지 예로 다음과 같이 CSV 파일을 곧바로 Spark DataFrame으로 읽어들일 수 있다.

```python
# Don't change this file path
file_path = "/usr/local/share/datasets/airports.csv"

# Read in the airports data
airports = spark.read.csv(file_path, header=True)

# Show the data
airports.show()
```
```
>>>
+---+--------------------+----------------+-----------------+----+---+---+
|faa|                name|             lat|              lon| alt| tz|dst|
+---+--------------------+----------------+-----------------+----+---+---+
|04G|   Lansdowne Airport|      41.1304722|      -80.6195833|1044| -5|  A|
|06A|Moton Field Munic...|      32.4605722|      -85.6800278| 264| -5|  A|
|06C| Schaumburg Regional|      41.9893408|      -88.1012428| 801| -6|  A|
|06N|     Randall Airport|       41.431912|      -74.3915611| 523| -5|  A|
|09J|Jekyll Island Air...|      31.0744722|      -81.4277778|  11| -4|  A|
|0A9|Elizabethton Muni...|      36.3712222|      -82.1734167|1593| -4|  A|
|0G6|Williams County A...|      41.4673056|      -84.5067778| 730| -5|  A|
|0G7|Finger Lakes Regi...|      42.8835647|      -76.7812318| 492| -5|  A|
|0P2|Shoestring Aviati...|      39.7948244|      -76.6471914|1000| -5|  U|
|0S9|Jefferson County ...|      48.0538086|     -122.8106436| 108| -8|  A|
|0W3|Harford County Ai...|      39.5668378|      -76.2024028| 409| -5|  A|
|10C|  Galt Field Airport|      42.4028889|      -88.3751111| 875| -6|  U|
|17G|Port Bucyrus-Craw...|      40.7815556|      -82.9748056|1003| -5|  A|
|19A|Jackson County Ai...|      34.1758638|      -83.5615972| 951| -4|  U|
|1A3|Martin Campbell F...|      35.0158056|      -84.3468333|1789| -4|  A|
|1B9| Mansfield Municipal|      42.0001331|      -71.1967714| 122| -5|  A|
|1C9|Frazier Lake Airpark|54.0133333333333|-124.768333333333| 152| -8|  A|
|1CS|Clow Internationa...|      41.6959744|      -88.1292306| 670| -6|  U|
|1G3|  Kent State Airport|      41.1513889|      -81.4151111|1134| -4|  A|
|1OH|     Fortman Airport|      40.5553253|      -84.3866186| 885| -5|  U|
+---+--------------------+----------------+-----------------+----+---+---+
only showing top 20 rows
```

### 2. Manipulating data

Spark의 DataFrame에서 정의하는 column-wise 명령어들을 알아보자. 특정 컬럼은 df.colName 과 같이 나타낼 수 있다. Spark DataFrame을 업데이트하는 일은 pandas와 다소 차이가 있다. Spark DataFrame은 immutable하다. 따라서, 항상 새로운 DataFrame을 리턴해야 한다.

#### .withColumn()

```python
# Create the DataFrame flights
flights = spark.table("flights")

# Add duration_hrs
flights = flights.withColumn("duration_hrs", flights.air_time/60)

# Show the head
flights.show()
```
```
>>>
+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+----+--------+--------+----+------+------------------+
|year|month|day|dep_time|dep_delay|arr_time|arr_delay|carrier|tailnum|flight|origin|dest|air_time|distance|hour|minute|      duration_hrs|
+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+----+--------+--------+----+------+------------------+
|2014|   12|  8|     658|       -7|     935|       -5|     VX| N846VA|  1780|   SEA| LAX|     132|     954|   6|    58|               2.2|
|2014|    1| 22|    1040|        5|    1505|        5|     AS| N559AS|   851|   SEA| HNL|     360|    2677|  10|    40|               6.0|
|2014|    3|  9|    1443|       -2|    1652|        2|     VX| N847VA|   755|   SEA| SFO|     111|     679|  14|    43|              1.85|
|2014|    4|  9|    1705|       45|    1839|       34|     WN| N360SW|   344|   PDX| SJC|      83|     569|  17|     5|1.3833333333333333|
|2014|    3|  9|     754|       -1|    1015|        1|     AS| N612AS|   522|   SEA| BUR|     127|     937|   7|    54|2.1166666666666667|
|2014|    1| 15|    1037|        7|    1352|        2|     WN| N646SW|    48|   PDX| DEN|     121|     991|  10|    37|2.0166666666666666|
|2014|    7|  2|     847|       42|    1041|       51|     WN| N422WN|  1520|   PDX| OAK|      90|     543|   8|    47|               1.5|
|2014|    5| 12|    1655|       -5|    1842|      -18|     VX| N361VA|   755|   SEA| SFO|      98|     679|  16|    55|1.6333333333333333|
|2014|    4| 19|    1236|       -4|    1508|       -7|     AS| N309AS|   490|   SEA| SAN|     135|    1050|  12|    36|              2.25|
|2014|   11| 19|    1812|       -3|    2352|       -4|     AS| N564AS|    26|   SEA| ORD|     198|    1721|  18|    12|               3.3|
|2014|   11|  8|    1653|       -2|    1924|       -1|     AS| N323AS|   448|   SEA| LAX|     130|     954|  16|    53|2.1666666666666665|
|2014|    8|  3|    1120|        0|    1415|        2|     AS| N305AS|   656|   SEA| PHX|     154|    1107|  11|    20| 2.566666666666667|
|2014|   10| 30|     811|       21|    1038|       29|     AS| N433AS|   608|   SEA| LAS|     127|     867|   8|    11|2.1166666666666667|
|2014|   11| 12|    2346|       -4|     217|      -28|     AS| N765AS|   121|   SEA| ANC|     183|    1448|  23|    46|              3.05|
|2014|   10| 31|    1314|       89|    1544|      111|     AS| N713AS|   306|   SEA| SFO|     129|     679|  13|    14|              2.15|
|2014|    1| 29|    2009|        3|    2159|        9|     UA| N27205|  1458|   PDX| SFO|      90|     550|  20|     9|               1.5|
|2014|   12| 17|    2015|       50|    2150|       41|     AS| N626AS|   368|   SEA| SMF|      76|     605|  20|    15|1.2666666666666666|
|2014|    8| 11|    1017|       -3|    1613|       -7|     WN| N8634A|   827|   SEA| MDW|     216|    1733|  10|    17|               3.6|
|2014|    1| 13|    2156|       -9|     607|      -15|     AS| N597AS|    24|   SEA| BOS|     290|    2496|  21|    56| 4.833333333333333|
|2014|    6|  5|    1733|      -12|    1945|      -10|     OO| N215AG|  3488|   PDX| BUR|     111|     817|  17|    33|              1.85|
+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+----+--------+--------+----+------+------------------+
only showing top 20 rows

```

위의 일은 다음과 같은 SQL 쿼리를 통해서 할 수 있다.

```sql
SELECT origin, dest, air_time / 60 FROM flights;
```

데이터를 chunk 단위로 쪼개고, 각 chunk 마다 summarizing을 할 수 있는 방법을 `GROUP BY`를 통해 할 수 있다. 즉, GROUP BY 로 묶은 컬럼의 unique value에 대해서 SQL 쿼리를 여러 번 실행하는 것과 같다. GROUP BY를 one column 이상으로 한다면, 각 column의 unique value들의 가능한 모든 조합에 대한 결과를 출력한다.

```sql
SELECT COUNT(*) FROM flights GROUP BY origin;

SELECT origin, dest, COUNT(*) FROM flights GROUP BY origin, dest;
```

#### .filter()

Spark DataFrame에서 `.filter()`라는 method가 있다. 이는 SQL의 WHERE과 같은 역할을 한다. `.filter()` method는 다음과 같이 두 가지 종류의 인자를 받는다. 

```python
# Filter flights by passing a string
long_flights1 = flights.filter("distance > 1000") (1) WHERE clause of a SQL query 

# Filter flights by passing a column of boolean values
long_flights2 = flights.filter(flights.distance > 1000) (2) a column of boolean values

# Print the data to check they're equal
long_flights1.show()
long_flights2.show()
```
```
>>>
+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+----+--------+--------+----+------+
|year|month|day|dep_time|dep_delay|arr_time|arr_delay|carrier|tailnum|flight|origin|dest|air_time|distance|hour|minute|
+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+----+--------+--------+----+------+
|2014|    1| 22|    1040|        5|    1505|        5|     AS| N559AS|   851|   SEA| HNL|     360|    2677|  10|    40|
|2014|    4| 19|    1236|       -4|    1508|       -7|     AS| N309AS|   490|   SEA| SAN|     135|    1050|  12|    36|
|2014|   11| 19|    1812|       -3|    2352|       -4|     AS| N564AS|    26|   SEA| ORD|     198|    1721|  18|    12|
|2014|    8|  3|    1120|        0|    1415|        2|     AS| N305AS|   656|   SEA| PHX|     154|    1107|  11|    20|
|2014|   11| 12|    2346|       -4|     217|      -28|     AS| N765AS|   121|   SEA| ANC|     183|    1448|  23|    46|
|2014|    8| 11|    1017|       -3|    1613|       -7|     WN| N8634A|   827|   SEA| MDW|     216|    1733|  10|    17|
|2014|    1| 13|    2156|       -9|     607|      -15|     AS| N597AS|    24|   SEA| BOS|     290|    2496|  21|    56|
|2014|    9| 26|     610|       -5|    1523|       65|     US| N127UW|   616|   SEA| PHL|     293|    2378|   6|    10|
|2014|   12|  4|     954|       -6|    1348|      -17|     HA| N395HA|    29|   SEA| OGG|     333|    2640|   9|    54|
|2014|    6|  4|    1115|        0|    1346|       -3|     AS| N461AS|   488|   SEA| SAN|     133|    1050|  11|    15|
|2014|    6| 26|    2054|       -1|    2318|       -6|     B6| N590JB|   907|   SEA| ANC|     179|    1448|  20|    54|
|2014|    6|  7|    1823|       -7|    2112|      -28|     AS| N512AS|   815|   SEA| LIH|     335|    2701|  18|    23|
|2014|    4| 30|     801|        1|    1757|       90|     AS| N407AS|    18|   SEA| MCO|     342|    2554|   8|     1|
|2014|   11| 29|     905|      155|    1655|      170|     DL| N824DN|  1598|   SEA| ATL|     229|    2182|   9|     5|
|2014|    6|  2|    2222|        7|      55|       15|     AS| N402AS|    99|   SEA| ANC|     190|    1448|  22|    22|
|2014|   11| 15|    1034|       -6|    1414|      -26|     AS| N589AS|   794|   SEA| ABQ|     139|    1180|  10|    34|
|2014|   10| 20|    1328|       -1|    1949|        4|     UA| N68805|  1212|   SEA| IAH|     228|    1874|  13|    28|
|2014|   12| 16|    1500|        0|    1906|       19|     US| N662AW|   500|   SEA| PHX|     151|    1107|  15|     0|
|2014|   11| 19|    1319|       -6|    1821|      -14|     DL| N309US|  2164|   PDX| MSP|     169|    1426|  13|    19|
|2014|    5| 21|     515|        0|     757|        0|     US| N172US|   593|   SEA| PHX|     143|    1107|   5|    15|
+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+----+--------+--------+----+------+
only showing top 20 rows

# same as above.
```

#### .select()

Spark에서 SQL의 SELECT 문은 `.select()` method로 할 수 있다. `.select()` method를 통해 `.withColumn()` method와 같이 덧셈이나 뺄셈 등의 operation을 수행할 수 있다. 그렇다면, `.select()`와 `.withColumn()` method의 차이점은 무엇인가? `.withColumn()`은 모든 컬럼을 리턴하고, `.select()`는 명시한 컬럼만 리턴한다. 데이터 wrangling 할 때, 매 step마다 원본은 구지 유지할 필요가 없다. 

```python
# Select the first set of columns
selected1 = flights.select("tailnum", "origin", "dest")

# Select the second set of columns
temp = flights.select(flights.origin, flights.dest, flights.carrier)

# Define first filter
filterA = flights.origin == "SEA"

# Define second filter
filterB = flights.dest == "PDX"

# Filter the data, first by filterA then by filterB
selected2 = temp.filter(filterA).filter(filterB)
```

`.select()` method를 SQL과 비슷하게, column-wise operation을 실행할 수 있다. `.selectExpr()` method를 통해 SQL string을 사용한다. `.alias()` method는 SQL의 AS와 같다.

```python
# Define avg_speed
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")

# Select the correct columns
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)

# Create the same table using a SQL expression
speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed")
```

#### Aggregating

보편적인 aggregation method인 `.min()`, `.max()`, 그리고 `.count()` 는 `GroupedData` method라 불린다. 이들은 `.groupBy()` DataFrame method를 부름과 동시에 생성된다. Aggregate를 하기 전에 groupby를 해줘야 한다. 여기 groupBy()에서는 인자가 없으므로 전체 범위라고 보면 된다.

```python
# Find the shortest flight from PDX in terms of distance
flights.filter(flights.origin == "PDX").groupBy().min("distance").show()

# Find the longest flight from SEA in terms of air time
flights.filter(flights.origin == "SEA").groupBy().max("air_time").show()
```
```
>>>
+-------------+
|min(distance)|
+-------------+
|          106|
+-------------+

+-------------+
|max(air_time)|
+-------------+
|          409|
+-------------+
```

```python
# Average duration of Delta flights
flights.filter(flights.carrier == "DL").filter(flights.origin == "SEA").groupBy().avg("air_time").show()

# Total hours in the air
flights.withColumn("duration_hrs", flights.air_time/60).groupBy().sum("duration_hrs").show()
```
```
>>>
+------------------+
|     avg(air_time)|
+------------------+
|188.20689655172413|
+------------------+

+------------------+
| sum(duration_hrs)|
+------------------+
|25289.600000000126|
+------------------+
```

#### Grouping and Aggregating

Aggregating 의 위력은 grouping과 같이 쓸 때 나온다. 지금까지는 `.groupBy()` method에 인자를 넣지 않았지만, column을 명시해서 `.groupBy()`를 실행해보자.

```python
# Group by tailnum
by_plane = flights.groupBy("tailnum")

# Number of flights each plane made
by_plane.count().show()

# Group by origin
by_origin = flights.groupBy("origin")

# Average duration of flights from PDX and SEA
by_origin.avg("air_time").show()
```
```
>>>
+-------+-----+
|tailnum|count|
+-------+-----+
| N442AS|   38|
| N102UW|    2|
| N36472|    4|
| N38451|    4|
| N73283|    4|
| N513UA|    2|
| N954WN|    5|
| N388DA|    3|
| N567AA|    1|
| N516UA|    2|
| N927DN|    1|
| N8322X|    1|
| N466SW|    1|
|  N6700|    1|
| N607AS|   45|
| N622SW|    4|
| N584AS|   31|
| N914WN|    4|
| N654AW|    2|
| N336NW|    1|
+-------+-----+
only showing top 20 rows

+------+------------------+
|origin|     avg(air_time)|
+------+------------------+
|   SEA| 160.4361496051259|
|   PDX|137.11543248288737|
+------+------------------+
```

`GroupedData` method 뿐만 아니라 `.agg()` method도 있다. `.agg()` method는 `pyspark.sql.functions` 서브 모듈로부터 aggregate function들을 사용하게 해준다. 해당 서브 모듈에 있는 모든 aggregation 함수들은 `GroupedData` 테이블에 있는 하나의 column을 인자로 받는다.

```python
# Import pyspark.sql.functions as F
import pyspark.sql.functions as F

# Group by month and dest
by_month_dest = flights.groupBy("month", "dest")

# Average departure delay by month and destination
by_month_dest.avg("dep_delay").show()

# Standard deviation of departure delay
by_month_dest.agg(F.stddev("dep_delay")).show()
```
```
>>>
+-----+----+--------------------+
|month|dest|      avg(dep_delay)|
+-----+----+--------------------+
|   11| TUS| -2.3333333333333335|
|   11| ANC|   7.529411764705882|
|    1| BUR|               -1.45|
|    1| PDX| -5.6923076923076925|
|    6| SBA|                -2.5|
|    5| LAX|-0.15789473684210525|
|   10| DTW|                 2.6|
|    6| SIT|                -1.0|
|   10| DFW|  18.176470588235293|
|    3| FAI|                -2.2|
|   10| SEA|                -0.8|
|    2| TUS| -0.6666666666666666|
|   12| OGG|  25.181818181818183|
|    9| DFW|   4.066666666666666|
|    5| EWR|               14.25|
|    3| RDM|                -6.2|
|    8| DCA|                 2.6|
|    7| ATL|   4.675675675675675|
|    4| JFK| 0.07142857142857142|
|   10| SNA| -1.1333333333333333|
+-----+----+--------------------+
only showing top 20 rows

+-----+----+----------------------+
|month|dest|stddev_samp(dep_delay)|
+-----+----+----------------------+
|   11| TUS|    3.0550504633038935|
|   11| ANC|    18.604716401245316|
|    1| BUR|     15.22627576540667|
|    1| PDX|     5.677214918493858|
|    6| SBA|     2.380476142847617|
|    5| LAX|     13.36268698685904|
|   10| DTW|     5.639148871948674|
|    6| SIT|                   NaN|
|   10| DFW|     45.53019017606675|
|    3| FAI|    3.1144823004794873|
|   10| SEA|     18.70523227029577|
|    2| TUS|    14.468356276140469|
|   12| OGG|     82.64480404939947|
|    9| DFW|    21.728629347782924|
|    5| EWR|     42.41595968929191|
|    3| RDM|      2.16794833886788|
|    8| DCA|     9.946523680831074|
|    7| ATL|    22.767001039582183|
|    4| JFK|     8.156774303176903|
|   10| SNA|    13.726234873756304|
+-----+----+----------------------+
only showing top 20 rows
```

#### Joining

```python
# Examine the data
airports.show()

# Rename the faa column
airports = airports.withColumnRenamed("faa", "dest")

# Join the DataFrames
flights_with_airports = flights.join(airports, on="dest", how="leftouter")

# Examine the new DataFrame
flights_with_airports.show()
```
```
>>>
+----+--------------------+----------------+-----------------+----+---+---+
|dest|                name|             lat|              lon| alt| tz|dst|
+----+--------------------+----------------+-----------------+----+---+---+
| 04G|   Lansdowne Airport|      41.1304722|      -80.6195833|1044| -5|  A|
| 06A|Moton Field Munic...|      32.4605722|      -85.6800278| 264| -5|  A|
| 06C| Schaumburg Regional|      41.9893408|      -88.1012428| 801| -6|  A|
| 06N|     Randall Airport|       41.431912|      -74.3915611| 523| -5|  A|
| 09J|Jekyll Island Air...|      31.0744722|      -81.4277778|  11| -4|  A|
| 0A9|Elizabethton Muni...|      36.3712222|      -82.1734167|1593| -4|  A|
| 0G6|Williams County A...|      41.4673056|      -84.5067778| 730| -5|  A|
| 0G7|Finger Lakes Regi...|      42.8835647|      -76.7812318| 492| -5|  A|
| 0P2|Shoestring Aviati...|      39.7948244|      -76.6471914|1000| -5|  U|
| 0S9|Jefferson County ...|      48.0538086|     -122.8106436| 108| -8|  A|
| 0W3|Harford County Ai...|      39.5668378|      -76.2024028| 409| -5|  A|
| 10C|  Galt Field Airport|      42.4028889|      -88.3751111| 875| -6|  U|
| 17G|Port Bucyrus-Craw...|      40.7815556|      -82.9748056|1003| -5|  A|
| 19A|Jackson County Ai...|      34.1758638|      -83.5615972| 951| -4|  U|
| 1A3|Martin Campbell F...|      35.0158056|      -84.3468333|1789| -4|  A|
| 1B9| Mansfield Municipal|      42.0001331|      -71.1967714| 122| -5|  A|
| 1C9|Frazier Lake Airpark|54.0133333333333|-124.768333333333| 152| -8|  A|
| 1CS|Clow Internationa...|      41.6959744|      -88.1292306| 670| -6|  U|
| 1G3|  Kent State Airport|      41.1513889|      -81.4151111|1134| -4|  A|
| 1OH|     Fortman Airport|      40.5553253|      -84.3866186| 885| -5|  U|
+----+--------------------+----------------+-----------------+----+---+---+
only showing top 20 rows

+----+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+--------+--------+----+------+--------------------+---------+-----------+----+---+---+
|dest|year|month|day|dep_time|dep_delay|arr_time|arr_delay|carrier|tailnum|flight|origin|air_time|distance|hour|minute|                name|      lat|        lon| alt| tz|dst|
+----+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+--------+--------+----+------+--------------------+---------+-----------+----+---+---+
| LAX|2014|   12|  8|     658|       -7|     935|       -5|     VX| N846VA|  1780|   SEA|     132|     954|   6|    58|    Los Angeles Intl|33.942536|-118.408075| 126| -8|  A|
| HNL|2014|    1| 22|    1040|        5|    1505|        5|     AS| N559AS|   851|   SEA|     360|    2677|  10|    40|       Honolulu Intl|21.318681|-157.922428|  13|-10|  N|
| SFO|2014|    3|  9|    1443|       -2|    1652|        2|     VX| N847VA|   755|   SEA|     111|     679|  14|    43|  San Francisco Intl|37.618972|-122.374889|  13| -8|  A|
| SJC|2014|    4|  9|    1705|       45|    1839|       34|     WN| N360SW|   344|   PDX|      83|     569|  17|     5|Norman Y Mineta S...|  37.3626|-121.929022|  62| -8|  A|
| BUR|2014|    3|  9|     754|       -1|    1015|        1|     AS| N612AS|   522|   SEA|     127|     937|   7|    54|            Bob Hope|34.200667|-118.358667| 778| -8|  A|
| DEN|2014|    1| 15|    1037|        7|    1352|        2|     WN| N646SW|    48|   PDX|     121|     991|  10|    37|         Denver Intl|39.861656|-104.673178|5431| -7|  A|
| OAK|2014|    7|  2|     847|       42|    1041|       51|     WN| N422WN|  1520|   PDX|      90|     543|   8|    47|Metropolitan Oakl...|37.721278|-122.220722|   9| -8|  A|
| SFO|2014|    5| 12|    1655|       -5|    1842|      -18|     VX| N361VA|   755|   SEA|      98|     679|  16|    55|  San Francisco Intl|37.618972|-122.374889|  13| -8|  A|
| SAN|2014|    4| 19|    1236|       -4|    1508|       -7|     AS| N309AS|   490|   SEA|     135|    1050|  12|    36|      San Diego Intl|32.733556|-117.189667|  17| -8|  A|
| ORD|2014|   11| 19|    1812|       -3|    2352|       -4|     AS| N564AS|    26|   SEA|     198|    1721|  18|    12|  Chicago Ohare Intl|41.978603| -87.904842| 668| -6|  A|
| LAX|2014|   11|  8|    1653|       -2|    1924|       -1|     AS| N323AS|   448|   SEA|     130|     954|  16|    53|    Los Angeles Intl|33.942536|-118.408075| 126| -8|  A|
| PHX|2014|    8|  3|    1120|        0|    1415|        2|     AS| N305AS|   656|   SEA|     154|    1107|  11|    20|Phoenix Sky Harbo...|33.434278|-112.011583|1135| -7|  N|
| LAS|2014|   10| 30|     811|       21|    1038|       29|     AS| N433AS|   608|   SEA|     127|     867|   8|    11|      Mc Carran Intl|36.080056| -115.15225|2141| -8|  A|
| ANC|2014|   11| 12|    2346|       -4|     217|      -28|     AS| N765AS|   121|   SEA|     183|    1448|  23|    46|Ted Stevens Ancho...|61.174361|-149.996361| 152| -9|  A|
| SFO|2014|   10| 31|    1314|       89|    1544|      111|     AS| N713AS|   306|   SEA|     129|     679|  13|    14|  San Francisco Intl|37.618972|-122.374889|  13| -8|  A|
| SFO|2014|    1| 29|    2009|        3|    2159|        9|     UA| N27205|  1458|   PDX|      90|     550|  20|     9|  San Francisco Intl|37.618972|-122.374889|  13| -8|  A|
| SMF|2014|   12| 17|    2015|       50|    2150|       41|     AS| N626AS|   368|   SEA|      76|     605|  20|    15|     Sacramento Intl|38.695417|-121.590778|  27| -8|  A|
| MDW|2014|    8| 11|    1017|       -3|    1613|       -7|     WN| N8634A|   827|   SEA|     216|    1733|  10|    17| Chicago Midway Intl|41.785972| -87.752417| 620| -6|  A|
| BOS|2014|    1| 13|    2156|       -9|     607|      -15|     AS| N597AS|    24|   SEA|     290|    2496|  21|    56|General Edward La...|42.364347| -71.005181|  19| -5|  A|
| BUR|2014|    6|  5|    1733|      -12|    1945|      -10|     OO| N215AG|  3488|   PDX|     111|     817|  17|    33|            Bob Hope|34.200667|-118.358667| 778| -8|  A|
+----+----+-----+---+--------+---------+--------+---------+-------+-------+------+------+--------+--------+----+------+--------------------+---------+-----------+----+---+---+
only showing top 20 rows
```

### 3. Getting started with machine learning pipelines

#### Machine Learning Pipelines

At the core of the `pyspark.ml` module are the `Transformer` and E`stimator` classes. Almost every other class in the module behaves similarly to these two basic classes.

`Transformer` classes have a `.transform()` method that takes a DataFrame and returns a new DataFrame; usually the original one with a new column appended. For example, you might use the class `Bucketizer` to create discrete bins from a continuous feature or the class `PCA` to reduce the dimensionality of your dataset using principal component analysis.

`Estimator` classes all implement a `.fit()` method. These methods also take a DataFrame, but instead of returning another DataFrame they return a model object. This can be something like a `StringIndexerModel` for including categorical data saved as strings in your models, or a `RandomForestModel` that uses the random forest algorithm for classification or regression.

#### Join the DataFrames

```python
# Rename year column
planes = planes.withColumnRenamed("year", "plane_year")

# Join the DataFrames
model_data = flights.join(planes, on="tailnum", how="leftouter")

model_data.show()

model_data.dtypes
```

```
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+
|tailnum|year|month|day|dep_time|dep_delay|arr_time|arr_delay|carrier|flight|origin|dest|air_time|distance|hour|minute|plane_year|                type|  manufacturer|      model|engines|seats|speed|   engine|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+
| N846VA|2014|   12|  8|     658|       -7|     935|       -5|     VX|  1780|   SEA| LAX|     132|     954|   6|    58|      2011|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|
| N559AS|2014|    1| 22|    1040|        5|    1505|        5|     AS|   851|   SEA| HNL|     360|    2677|  10|    40|      2006|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|
| N847VA|2014|    3|  9|    1443|       -2|    1652|        2|     VX|   755|   SEA| SFO|     111|     679|  14|    43|      2011|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|
| N360SW|2014|    4|  9|    1705|       45|    1839|       34|     WN|   344|   PDX| SJC|      83|     569|  17|     5|      1992|Fixed wing multi ...|        BOEING|    737-3H4|      2|  149|   NA|Turbo-fan|
| N612AS|2014|    3|  9|     754|       -1|    1015|        1|     AS|   522|   SEA| BUR|     127|     937|   7|    54|      1999|Fixed wing multi ...|        BOEING|    737-790|      2|  151|   NA|Turbo-jet|
| N646SW|2014|    1| 15|    1037|        7|    1352|        2|     WN|    48|   PDX| DEN|     121|     991|  10|    37|      1997|Fixed wing multi ...|        BOEING|    737-3H4|      2|  149|   NA|Turbo-fan|
| N422WN|2014|    7|  2|     847|       42|    1041|       51|     WN|  1520|   PDX| OAK|      90|     543|   8|    47|      2002|Fixed wing multi ...|        BOEING|    737-7H4|      2|  140|   NA|Turbo-fan|
| N361VA|2014|    5| 12|    1655|       -5|    1842|      -18|     VX|   755|   SEA| SFO|      98|     679|  16|    55|      2013|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|
| N309AS|2014|    4| 19|    1236|       -4|    1508|       -7|     AS|   490|   SEA| SAN|     135|    1050|  12|    36|      2001|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|
| N564AS|2014|   11| 19|    1812|       -3|    2352|       -4|     AS|    26|   SEA| ORD|     198|    1721|  18|    12|      2006|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|
| N323AS|2014|   11|  8|    1653|       -2|    1924|       -1|     AS|   448|   SEA| LAX|     130|     954|  16|    53|      2004|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|
| N305AS|2014|    8|  3|    1120|        0|    1415|        2|     AS|   656|   SEA| PHX|     154|    1107|  11|    20|      2001|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|
| N433AS|2014|   10| 30|     811|       21|    1038|       29|     AS|   608|   SEA| LAS|     127|     867|   8|    11|      2013|Fixed wing multi ...|        BOEING|  737-990ER|      2|  222|   NA|Turbo-fan|
| N765AS|2014|   11| 12|    2346|       -4|     217|      -28|     AS|   121|   SEA| ANC|     183|    1448|  23|    46|      1992|Fixed wing multi ...|        BOEING|    737-4Q8|      2|  149|   NA|Turbo-fan|
| N713AS|2014|   10| 31|    1314|       89|    1544|      111|     AS|   306|   SEA| SFO|     129|     679|  13|    14|      1999|Fixed wing multi ...|        BOEING|    737-490|      2|  149|   NA|Turbo-jet|
| N27205|2014|    1| 29|    2009|        3|    2159|        9|     UA|  1458|   PDX| SFO|      90|     550|  20|     9|      2000|Fixed wing multi ...|        BOEING|    737-824|      2|  149|   NA|Turbo-fan|
| N626AS|2014|   12| 17|    2015|       50|    2150|       41|     AS|   368|   SEA| SMF|      76|     605|  20|    15|      2001|Fixed wing multi ...|        BOEING|    737-790|      2|  151|   NA|Turbo-jet|
| N8634A|2014|    8| 11|    1017|       -3|    1613|       -7|     WN|   827|   SEA| MDW|     216|    1733|  10|    17|      2014|Fixed wing multi ...|        BOEING|    737-8H4|      2|  140|   NA|Turbo-fan|
| N597AS|2014|    1| 13|    2156|       -9|     607|      -15|     AS|    24|   SEA| BOS|     290|    2496|  21|    56|      2008|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|
| N215AG|2014|    6|  5|    1733|      -12|    1945|      -10|     OO|  3488|   PDX| BUR|     111|     817|  17|    33|      2001|Fixed wing multi ...|BOMBARDIER INC|CL-600-2C10|      2|   80|   NA|Turbo-fan|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+
only showing top 20 rows

[('tailnum', 'string'),
 ('year', 'string'),
 ('month', 'string'),
 ('day', 'string'),
 ('dep_time', 'string'),
 ('dep_delay', 'string'),
 ('arr_time', 'string'),
 ('arr_delay', 'string'),
 ('carrier', 'string'),
 ('flight', 'string'),
 ('origin', 'string'),
 ('dest', 'string'),
 ('air_time', 'string'),
 ('distance', 'string'),
 ('hour', 'string'),
 ('minute', 'string'),
 ('plane_year', 'string'),
 ('type', 'string'),
 ('manufacturer', 'string'),
 ('model', 'string'),
 ('engines', 'string'),
 ('seats', 'string'),
 ('speed', 'string'),
 ('engine', 'string')]
```

#### Data types

Spark only handles numeric data. That means all of the columns in your DataFrame must be either integers or decimals (called 'doubles' in Spark).

You can see that some of the columns in our DataFrame are strings containing numbers as opposed to actual numeric values.

To remedy this, you can use the `.cast()` method in combination with the `.withColumn()` method. It's important to note that `.cast()` works on columns, while `.withColumn()` works on DataFrames.

The only argument you need to pass to `.cast()` is the kind of value you want to create, in string form. For example, to create integers, you'll pass the argument `"integer"` and for decimal numbers you'll use `"double"`.

#### String to integer

```python
# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast("integer"))
model_data = model_data.withColumn("air_time", model_data.air_time.cast("integer"))
model_data = model_data.withColumn("month", model_data.month.cast("integer"))
model_data = model_data.withColumn("plane_year", model_data.plane_year.cast("integer"))

model_data.dtypes
```
```
[('tailnum', 'string'),
 ('year', 'string'),
 ('month', 'int'),
 ('day', 'string'),
 ('dep_time', 'string'),
 ('dep_delay', 'string'),
 ('arr_time', 'string'),
 ('arr_delay', 'int'),
 ('carrier', 'string'),
 ('flight', 'string'),
 ('origin', 'string'),
 ('dest', 'string'),
 ('air_time', 'int'),
 ('distance', 'string'),
 ('hour', 'string'),
 ('minute', 'string'),
 ('plane_year', 'int'),
 ('type', 'string'),
 ('manufacturer', 'string'),
 ('model', 'string'),
 ('engines', 'string'),
 ('seats', 'string'),
 ('speed', 'string'),
 ('engine', 'string')]
```

#### Create a new column

```python
# Create the column plane_age
model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)
```

#### Making a Boolean

```python
# Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)

# Convert the boolean column to an integer.
model_data = model_data.withColumn("label", model_data.is_late.cast("integer"))

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")
```

#### Strings and factors

The airline and the plane's destination as features in your model. These are coded as strings and there isn't any obvious way to convert them to a numeric data type.

Fortunately, PySpark has functions for handling this built into the `pyspark.ml.features` submodule. You can create what are called 'one-hot vectors' to represent the carrier and the destination of each flight. 

The first step to encoding your categorical feature is to create a `StringIndexer`. Members of this class are `Estimator`s that take a DataFrame with a column of strings and map each unique string to a number. Then, the `Estimator` returns a `Transformer` that takes a DataFrame, attaches the mapping to it as metadata, and returns a new DataFrame with a numeric column corresponding to the string column.

The second step is to encode this numeric column as a one-hot vector using a `OneHotEncoder`. This works exactly the same way as the `StringIndexer` by creating an `Estimator` and then a `Transformer`. The end result is a column that encodes your categorical feature as a vector that's suitable for machine learning routines!

##### Carrier

```python
# Create a StringIndexer
carr_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_index") 
# StringIndexer(): Estimator
# carr_indexer: Transformer

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol="carrier_index", outputCol="carrier_fact")
# OneHotEncoder(): Estimator
# carr_encoder: Transformer
```

##### Destination

```python
# Create a StringIndexer
dest_indexer = StringIndexer(inputCol="dest", outputCol="dest_index")

# Create a OneHotEncoder
dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol="dest_fact")
```

#### Assemble a vector

The last step in the `Pipeline` is to combine all of the columns containing our features into a single column. This has to be done before modeling can take place because every Spark modeling routine expects the data to be in this form. You can do this by storing each of the values from a column as an entry in a vector. Then, from the model's point of view, every observation is a vector that contains all of the information about it and a label that tells the modeler what value that observation corresponds to.

```python
# Make a VectorAssembler
# VectorAssembler: This Transformer takes all of the columns you specify and combines them into a new vector column.
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")
```

#### Create the pipeline

`Pipeline` is a class in the `pyspark.ml` module that combines all the `Estimator`s and `Transformer`s that you've already created. This lets you reuse the same modeling process over and over again by wrapping it up in one simple object. Neat, right?

```python
# Import Pipeline
from pyspark.ml import Pipeline

# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])
```

#### Test vs. Train

In Spark it's important to make sure you split the data after all the transformations. This is because operations like `StringIndexer` don't always produce the same index even when given the same list of strings.

#### Transform the data

```python
# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)
```

#### Split the data

```python
# Split the data into training and test sets
training, test = piped_data.randomSplit([.6, .4])
```

### 4. Model tuning and selection

#### What is logistic regression?

#### Create the modeler

```python
# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression

# Create a LogisticRegression Estimator
lr = LogisticRegression()
```

#### Cross validation

You'll be using cross validation to choose the hyperparameters by creating a grid of the possible pairs of values for the two hyperparameters, `elasticNetParam` and `regParam`, and using the cross validation error to compare all the different models so you can choose the best one!

#### Create the evaluator

```python
# Import the evaluation submodule
import pyspark.ml.evaluation as evals

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")
```

#### Make a grid

```python
# Import the tuning submodule
import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# Build the grid
grid = grid.build()
```

#### Make the validator

```python
# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
                         estimatorParamMaps=grid,
                         evaluator=evaluator
                         )
```

#### Fit the model(s)

```python
# Fit cross validation models
# Call lr.fit()
best_lr = lr.fit(training)

# Extract the best model
# Print best_lr
print(best_lr)
```
```
LogisticRegression_4e4baee65b27b86405e6
```

#### Evaluating binary classifiers

For this course we'll be using a common metric for binary classification algorithms call the AUC, or area under the curve. In this case, the curve is the ROC, or receiver operating curve. The details of what these things actually measure isn't important for this course. All you need to know is that for our purposes, the closer the AUC is to one (1), the better the model is!

#### Evaluate the model

```python
# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))
```
```
0.7125950520012989
```





<br>

## Big Data Fundamentals with PySpark

### 2. Programming in PySpark RDD’s

RDDs (Resilient Distributed Datasets)

* Resilient: ability to withstand failures
* Distributed: spanning across multiple machines
* Datasets: collection of partitioned data (e.g. arrays, tables, tuples etc.)

Creating RDDs

* Parallelizing an existing collection of objects
* External datasets (files in HDFS, Objects in Amazon S3, lines in a text file)
* From existing RDDs

Understanding Partitioning in PySpark

* A partition is a logical division of a large distributed dataset

#### RDDs from Parallelized collections

RDD는 Spark에서 가장 기본적인 추상화 자료 구조이다. 이는 immutable distributed collection of objects이다. 

```python
# Create an RDD from a list of words
RDD = sc.parallelize(["Spark", "is", "a", "framework", "for", "Big Data processing"])

# Print out the type of the created object
print("The type of RDD is", type(RDD))
```
```
>>>
The type of RDD is <class 'pyspark.rdd.RDD'>
```

#### RDDs from External Datasets

```python
# Print the file_path
print("The file_path is", file_path)

# Create a fileRDD from file_path
fileRDD = sc.textFile(file_path)

# Check the type of fileRDD
print("The file type of fileRDD is", type(fileRDD))
```
```
>>>
The file_path is /usr/local/share/datasets/README.md
The file type of fileRDD is <class 'pyspark.rdd.RDD'>
```











### 3. PySpark SQL & DataFrames

PySpark DataFrames

* PySpark SQL은 structured data를 위한 Spark library이다. 
* PySpark DataFrame은 immutable하고, distributed collection of data에 column 네이밍을 붙인 형태이다.
* structured data(e.g. relational database)와 semi-structured data(e.g. Json) 모두 처리 가능하게 디자인되었다.
* DataFrame API는 Python, R, Scala, Java에서 사용할 수 있다.
* DataFrame은 SQL 쿼리(e.g. SELECT .. from table) 또는 expression methods (e.g. df.select())를 모두 지원한다.

SparkSession - Entry point for DataFrame API

* SparkContext는 RDDs를 생성하는데 main entry point이다.
* SparkSession은 Spark DataFrames와 상호 작용하기 위해서 제공되는 a single point of entry이다.
* SparkSession은 DataFrame을 생성하기 위해, 등록하기 위해, 그리고 SQL 쿼리를 실행하기 위해 사용된다.
* SparkSession은 PySpark shell에서 spark로 사용된다.

Creating DataFrames in PySpark

* Pyspark DataFrame을 만드는 방법은 2가지가 있다.
   * 기존 존재하는 RDDs 에서... SparkSession의 createDataFrame() method를 사용한다.
   * 다양한 데이터 소스(ex. CSV, JSON, TXT)에서... SparkSession의 read method를 사용한다.
* Schema는 data를 컨트롤하고, DataFrame이 쿼리를 최적화하는 데 도움을 준다.
* Schema는 column name, type of data, empty value 등의 정보를 제공한다.


#### RDD to DataFrame

Spark에서 RDDs가 더 본질적인 자료 구조이지만, DataFrame이 다루기 더 쉽다.

```python
# Create a list of tuples
sample_list = [('Mona',20), ('Jennifer',34), ('John',20), ('Jim',26)]

# Create a RDD from the list
rdd = sc.parallelize(sample_list)

# Create a PySpark DataFrame
names_df = spark.createDataFrame(rdd, schema=['Name', 'Age'])

# Check the type of names_df
print("The type of names_df is", type(names_df))
>>>
The type of names_df is <class 'pyspark.sql.dataframe.DataFrame'>
```

#### Loading CSV into DataFrame

```python
# Create an DataFrame from file_path
people_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Check the type of people_df
print("The type of people_df is", type(people_df))
>>>
The type of people_df is <class 'pyspark.sql.dataframe.DataFrame'>
```

DataFrame operators in PySpark

* DataFrame operations: Transformations and Actions
* DataFrame Transformations:
   * select(), filter(), groupby(), orderby(), dropDuplicates(), and withColumnRenamed()
* DataFrame Actions:
   * printSchema(), head(), show(), count(), columns and describe()
   
#### Inspecting data in PySpark DataFrame 

```python
# Print the first 10 observations 
people_df.show(10)

# Count the number of rows 
print("There are {} rows in the people_df DataFrame.".format(people_df.count()))

# Count the number of columns and their names 
print("There are {} columns in the people_df DataFrame and their names are {}".format(len(people_df.columns), people_df.columns))
```
```
>>>
+---+---------+----------------+------+-------------+
|_c0|person_id|            name|   sex|date of birth|
+---+---------+----------------+------+-------------+
|  0|      100|  Penelope Lewis|female|   1990-08-31|
|  1|      101|   David Anthony|  male|   1971-10-14|
|  2|      102|       Ida Shipp|female|   1962-05-24|
|  3|      103|    Joanna Moore|female|   2017-03-10|
|  4|      104|  Lisandra Ortiz|female|   2020-08-05|
|  5|      105|   David Simmons|  male|   1999-12-30|
|  6|      106|   Edward Hudson|  male|   1983-05-09|
|  7|      107|    Albert Jones|  male|   1990-09-13|
|  8|      108|Leonard Cavender|  male|   1958-08-08|
|  9|      109|  Everett Vadala|  male|   2005-05-24|
+---+---------+----------------+------+-------------+
only showing top 10 rows

There are 100000 rows in the people_df DataFrame.
There are 5 columns in the people_df DataFrame and their names are ['_c0', 'person_id', 'name', 'sex', 'date of birth']
```

#### PySpark DataFrame subsetting and cleaning

```python
# Select name, sex and date of birth columns
people_df_sub = people_df.select('name', 'sex', 'date of birth')

# Print the first 10 observations from people_df_sub
people_df_sub.show(10)

# Remove duplicate entries from people_df_sub
people_df_sub_nodup = people_df_sub.dropDuplicates()

# Count the number of rows
print("There were {} rows before removing duplicates, and {} rows after removing duplicates".format(people_df_sub.count(), people_df_sub_nodup.count()))
```
```
>>>
+----------------+------+-------------+
|            name|   sex|date of birth|
+----------------+------+-------------+
|  Penelope Lewis|female|   1990-08-31|
|   David Anthony|  male|   1971-10-14|
|       Ida Shipp|female|   1962-05-24|
|    Joanna Moore|female|   2017-03-10|
|  Lisandra Ortiz|female|   2020-08-05|
|   David Simmons|  male|   1999-12-30|
|   Edward Hudson|  male|   1983-05-09|
|    Albert Jones|  male|   1990-09-13|
|Leonard Cavender|  male|   1958-08-08|
|  Everett Vadala|  male|   2005-05-24|
+----------------+------+-------------+
only showing top 10 rows

There were 100000 rows before removing duplicates, and 99998 rows after removing duplicates
```

#### Filtering your DataFrame

`select()`는 DataFrame을 column-wise로 subset한다. `filter()`는 DataFrame을 어떤 condition에 기반해서 row-wise로 subset한다.

```python
# Filter people_df to select females 
people_df_female = people_df.filter(people_df.sex == "female")

# Filter people_df to select males
people_df_male = people_df.filter(people_df.sex == "male")

# Count the number of rows 
print("There are {} rows in the people_df_female DataFrame and {} rows in the people_df_male DataFrame".format(people_df_female.count(), people_df_male.count()))

>>>
There are 49014 rows in the people_df_female DataFrame and 49066 rows in the people_df_male DataFrame
```

#### Running SQL Queries Programmatically

The `sql()` function on a SparkSession enables applications to run SQL queries programmatically and returns the result as another DataFrame. 

```python
# Create a temporary table "people"
people_df.createOrReplaceTempView("people")

# Construct a query to select the names of the people from the temporary table "people"
query = '''SELECT name FROM people'''

# Assign the result of Spark's query to people_df_names
people_df_names = spark.sql(query)

# Print the top 10 names of the people
people_df_names.show(10)
```
```
>>>
+----------------+
|            name|
+----------------+
|  Penelope Lewis|
|   David Anthony|
|       Ida Shipp|
|    Joanna Moore|
|  Lisandra Ortiz|
|   David Simmons|
|   Edward Hudson|
|    Albert Jones|
|Leonard Cavender|
|  Everett Vadala|
+----------------+
only showing top 10 rows
```

#### SQL queries for filtering Table

```python
# Filter the people table to select female sex 
people_female_df = spark.sql('SELECT * FROM people WHERE sex=="female"')

# Filter the people table DataFrame to select male sex
people_male_df = spark.sql('SELECT * FROM people WHERE sex=="male"')

# Count the number of rows in both people_df_female and people_male_df DataFrames
print("There are {} rows in the people_female_df and {} rows in the people_male_df DataFrames".format(people_female_df.count(), people_male_df.count()))
>>>
There are 49014 rows in the people_female_df and 49066 rows in the people_male_df DataFrames
```

#### PySpark DataFrame visualization

```python
# Check the column names of names_df
print("The column names of names_df are", names_df.columns)

# Convert to Pandas DataFrame  
df_pandas = names_df.toPandas()

# Create a horizontal bar plot
df_pandas.plot(kind='barh', x='Name', y='Age', colormap='winter_r')
plt.show()

>>>
...
```

#### EDA on the "FIFA 2018 WORLD

##### Part 1: Create a DataFrame from CSV file

```python
# Load the Dataframe 
fifa_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Check the schema of columns 
fifa_df.printSchema()

# Show the first 10 observations
fifa_df.show(10)

# Print the total number of rows
print("There are {} rows in the fifa_df DataFrame".format(fifa_df.count()))
```
```
>>>
root
 |-- _c0: integer (nullable = true)
 |-- Name: string (nullable = true)
 |-- Age: integer (nullable = true)
 |-- Photo: string (nullable = true)
 |-- Nationality: string (nullable = true)
 |-- Flag: string (nullable = true)
 |-- Overall: integer (nullable = true)
 |-- Potential: integer (nullable = true)
 |-- Club: string (nullable = true)
 |-- Club Logo: string (nullable = true)
 |-- Value: string (nullable = true)
 |-- Wage: string (nullable = true)
 |-- Special: integer (nullable = true)
 |-- Acceleration: string (nullable = true)
 |-- Aggression: string (nullable = true)
 |-- Agility: string (nullable = true)
 |-- Balance: string (nullable = true)
 |-- Ball control: string (nullable = true)
 |-- Composure: string (nullable = true)
 |-- Crossing: string (nullable = true)
 |-- Curve: string (nullable = true)
 |-- Dribbling: string (nullable = true)
 |-- Finishing: string (nullable = true)
 |-- Free kick accuracy: string (nullable = true)
 |-- GK diving: string (nullable = true)
 |-- GK handling: string (nullable = true)
 |-- GK kicking: string (nullable = true)
 |-- GK positioning: string (nullable = true)
 |-- GK reflexes: string (nullable = true)
 |-- Heading accuracy: string (nullable = true)
 |-- Interceptions: string (nullable = true)
 |-- Jumping: string (nullable = true)
 |-- Long passing: string (nullable = true)
 |-- Long shots: string (nullable = true)
 |-- Marking: string (nullable = true)
 |-- Penalties: string (nullable = true)
 |-- Positioning: string (nullable = true)
 |-- Reactions: string (nullable = true)
 |-- Short passing: string (nullable = true)
 |-- Shot power: string (nullable = true)
 |-- Sliding tackle: string (nullable = true)
 |-- Sprint speed: string (nullable = true)
 |-- Stamina: string (nullable = true)
 |-- Standing tackle: string (nullable = true)
 |-- Strength: string (nullable = true)
 |-- Vision: string (nullable = true)
 |-- Volleys: string (nullable = true)
 |-- CAM: double (nullable = true)
 |-- CB: double (nullable = true)
 |-- CDM: double (nullable = true)
 |-- CF: double (nullable = true)
 |-- CM: double (nullable = true)
 |-- ID: integer (nullable = true)
 |-- LAM: double (nullable = true)
 |-- LB: double (nullable = true)
 |-- LCB: double (nullable = true)
 |-- LCM: double (nullable = true)
 |-- LDM: double (nullable = true)
 |-- LF: double (nullable = true)
 |-- LM: double (nullable = true)
 |-- LS: double (nullable = true)
 |-- LW: double (nullable = true)
 |-- LWB: double (nullable = true)
 |-- Preferred Positions: string (nullable = true)
 |-- RAM: double (nullable = true)
 |-- RB: double (nullable = true)
 |-- RCB: double (nullable = true)
 |-- RCM: double (nullable = true)
 |-- RDM: double (nullable = true)
 |-- RF: double (nullable = true)
 |-- RM: double (nullable = true)
 |-- RS: double (nullable = true)
 |-- RW: double (nullable = true)
 |-- RWB: double (nullable = true)
 |-- ST: double (nullable = true)

+---+-----------------+---+--------------------+-----------+--------------------+-------+---------+-------------------+--------------------+------+-----+-------+------------+----------+-------+-------+------------+---------+--------+-----+---------+---------+------------------+---------+-----------+----------+--------------+-----------+----------------+-------------+-------+------------+----------+-------+---------+-----------+---------+-------------+----------+--------------+------------+-------+---------------+--------+------+-------+----+----+----+----+----+------+----+----+----+----+----+----+----+----+----+----+-------------------+----+----+----+----+----+----+----+----+----+----+----+
|_c0|             Name|Age|               Photo|Nationality|                Flag|Overall|Potential|               Club|           Club Logo| Value| Wage|Special|Acceleration|Aggression|Agility|Balance|Ball control|Composure|Crossing|Curve|Dribbling|Finishing|Free kick accuracy|GK diving|GK handling|GK kicking|GK positioning|GK reflexes|Heading accuracy|Interceptions|Jumping|Long passing|Long shots|Marking|Penalties|Positioning|Reactions|Short passing|Shot power|Sliding tackle|Sprint speed|Stamina|Standing tackle|Strength|Vision|Volleys| CAM|  CB| CDM|  CF|  CM|    ID| LAM|  LB| LCB| LCM| LDM|  LF|  LM|  LS|  LW| LWB|Preferred Positions| RAM|  RB| RCB| RCM| RDM|  RF|  RM|  RS|  RW| RWB|  ST|
+---+-----------------+---+--------------------+-----------+--------------------+-------+---------+-------------------+--------------------+------+-----+-------+------------+----------+-------+-------+------------+---------+--------+-----+---------+---------+------------------+---------+-----------+----------+--------------+-----------+----------------+-------------+-------+------------+----------+-------+---------+-----------+---------+-------------+----------+--------------+------------+-------+---------------+--------+------+-------+----+----+----+----+----+------+----+----+----+----+----+----+----+----+----+----+-------------------+----+----+----+----+----+----+----+----+----+----+----+
|  0|Cristiano Ronaldo| 32|https://cdn.sofif...|   Portugal|https://cdn.sofif...|     94|       94|     Real Madrid CF|https://cdn.sofif...|€95.5M|€565K|   2228|          89|        63|     89|     63|          93|       95|      85|   81|       91|       94|                76|        7|         11|        15|            14|         11|              88|           29|     95|          77|        92|     22|       85|         95|       96|           83|        94|            23|          91|     92|             31|      80|    85|     88|89.0|53.0|62.0|91.0|82.0| 20801|89.0|61.0|53.0|82.0|62.0|91.0|89.0|92.0|91.0|66.0|             ST LW |89.0|61.0|53.0|82.0|62.0|91.0|89.0|92.0|91.0|66.0|92.0|
|  1|         L. Messi| 30|https://cdn.sofif...|  Argentina|https://cdn.sofif...|     93|       93|       FC Barcelona|https://cdn.sofif...| €105M|€565K|   2154|          92|        48|     90|     95|          95|       96|      77|   89|       97|       95|                90|        6|         11|        15|            14|          8|              71|           22|     68|          87|        88|     13|       74|         93|       95|           88|        85|            26|          87|     73|             28|      59|    90|     85|92.0|45.0|59.0|92.0|84.0|158023|92.0|57.0|45.0|84.0|59.0|92.0|90.0|88.0|91.0|62.0|                RW |92.0|57.0|45.0|84.0|59.0|92.0|90.0|88.0|91.0|62.0|88.0|
|  2|           Neymar| 25|https://cdn.sofif...|     Brazil|https://cdn.sofif...|     92|       94|Paris Saint-Germain|https://cdn.sofif...| €123M|€280K|   2100|          94|        56|     96|     82|          95|       92|      75|   81|       96|       89|                84|        9|          9|        15|            15|         11|              62|           36|     61|          75|        77|     21|       81|         90|       88|           81|        80|            33|          90|     78|             24|      53|    80|     83|88.0|46.0|59.0|88.0|79.0|190871|88.0|59.0|46.0|79.0|59.0|88.0|87.0|84.0|89.0|64.0|                LW |88.0|59.0|46.0|79.0|59.0|88.0|87.0|84.0|89.0|64.0|84.0|
|  3|        L. Suárez| 30|https://cdn.sofif...|    Uruguay|https://cdn.sofif...|     92|       92|       FC Barcelona|https://cdn.sofif...|  €97M|€510K|   2291|          88|        78|     86|     60|          91|       83|      77|   86|       86|       94|                84|       27|         25|        31|            33|         37|              77|           41|     69|          64|        86|     30|       85|         92|       93|           83|        87|            38|          77|     89|             45|      80|    84|     88|87.0|58.0|65.0|88.0|80.0|176580|87.0|64.0|58.0|80.0|65.0|88.0|85.0|88.0|87.0|68.0|                ST |87.0|64.0|58.0|80.0|65.0|88.0|85.0|88.0|87.0|68.0|88.0|
|  4|         M. Neuer| 31|https://cdn.sofif...|    Germany|https://cdn.sofif...|     92|       92|   FC Bayern Munich|https://cdn.sofif...|  €61M|€230K|   1493|          58|        29|     52|     35|          48|       70|      15|   14|       30|       13|                11|       91|         90|        95|            91|         89|              25|           30|     78|          59|        16|     10|       47|         12|       85|           55|        25|            11|          61|     44|             10|      83|    70|     11|null|null|null|null|null|167495|null|null|null|null|null|null|null|null|null|null|                GK |null|null|null|null|null|null|null|null|null|null|null|
|  5|   R. Lewandowski| 28|https://cdn.sofif...|     Poland|https://cdn.sofif...|     91|       91|   FC Bayern Munich|https://cdn.sofif...|  €92M|€355K|   2143|          79|        80|     78|     80|          89|       87|      62|   77|       85|       91|                84|       15|          6|        12|             8|         10|              85|           39|     84|          65|        83|     25|       81|         91|       91|           83|        88|            19|          83|     79|             42|      84|    78|     87|84.0|57.0|62.0|87.0|78.0|188545|84.0|58.0|57.0|78.0|62.0|87.0|82.0|88.0|84.0|61.0|                ST |84.0|58.0|57.0|78.0|62.0|87.0|82.0|88.0|84.0|61.0|88.0|
|  6|           De Gea| 26|https://cdn.sofif...|      Spain|https://cdn.sofif...|     90|       92|  Manchester United|https://cdn.sofif...|€64.5M|€215K|   1458|          57|        38|     60|     43|          42|       64|      17|   21|       18|       13|                19|       90|         85|        87|            86|         90|              21|           30|     67|          51|        12|     13|       40|         12|       88|           50|        31|            13|          58|     40|             21|      64|    68|     13|null|null|null|null|null|193080|null|null|null|null|null|null|null|null|null|null|                GK |null|null|null|null|null|null|null|null|null|null|null|
|  7|        E. Hazard| 26|https://cdn.sofif...|    Belgium|https://cdn.sofif...|     90|       91|            Chelsea|https://cdn.sofif...|€90.5M|€295K|   2096|          93|        54|     93|     91|          92|       87|      80|   82|       93|       83|                79|       11|         12|         6|             8|          8|              57|           41|     59|          81|        82|     25|       86|         85|       85|           86|        79|            22|          87|     79|             27|      65|    86|     79|88.0|47.0|61.0|87.0|81.0|183277|88.0|59.0|47.0|81.0|61.0|87.0|87.0|82.0|88.0|64.0|                LW |88.0|59.0|47.0|81.0|61.0|87.0|87.0|82.0|88.0|64.0|82.0|
|  8|         T. Kroos| 27|https://cdn.sofif...|    Germany|https://cdn.sofif...|     90|       90|     Real Madrid CF|https://cdn.sofif...|  €79M|€340K|   2165|          60|        60|     71|     69|          89|       85|      85|   85|       79|       76|                84|       10|         11|        13|             7|         10|              54|           85|     32|          93|        90|     63|       73|         79|       86|           90|        87|            69|          52|     77|             82|      74|    88|     82|83.0|72.0|82.0|81.0|87.0|182521|83.0|76.0|72.0|87.0|82.0|81.0|81.0|77.0|80.0|78.0|            CDM CM |83.0|76.0|72.0|87.0|82.0|81.0|81.0|77.0|80.0|78.0|77.0|
|  9|       G. Higuaín| 29|https://cdn.sofif...|  Argentina|https://cdn.sofif...|     90|       90|           Juventus|https://cdn.sofif...|  €77M|€275K|   1961|          78|        50|     75|     69|          85|       86|      68|   74|       84|       91|                62|        5|         12|         7|             5|         10|              86|           20|     79|          59|        82|     12|       70|         92|       88|           75|        88|            18|          80|     72|             22|      85|    70|     88|81.0|46.0|52.0|84.0|71.0|167664|81.0|51.0|46.0|71.0|52.0|84.0|79.0|87.0|82.0|55.0|                ST |81.0|51.0|46.0|71.0|52.0|84.0|79.0|87.0|82.0|55.0|87.0|
+---+-----------------+---+--------------------+-----------+--------------------+-------+---------+-------------------+--------------------+------+-----+-------+------------+----------+-------+-------+------------+---------+--------+-----+---------+---------+------------------+---------+-----------+----------+--------------+-----------+----------------+-------------+-------+------------+----------+-------+---------+-----------+---------+-------------+----------+--------------+------------+-------+---------------+--------+------+-------+----+----+----+----+----+------+----+----+----+----+----+----+----+----+----+----+-------------------+----+----+----+----+----+----+----+----+----+----+----+
only showing top 10 rows

There are 17981 rows in the fifa_df DataFrame

```

##### Part 2: SQL Queries on DataFrame

```python
# Create a temporary view of fifa_df
fifa_df.createOrReplaceTempView('fifa_df_table')

# Construct the "query"
query = '''SELECT Age FROM fifa_df_table WHERE Nationality == "Germany"'''

# Apply the SQL "query"
fifa_df_germany_age = spark.sql(query)

# Generate basic statistics
fifa_df_germany_age.describe().show()
```
```
>>>
+-------+-----------------+
|summary|              Age|
+-------+-----------------+
|  count|             1140|
|   mean|24.20263157894737|
| stddev|4.197096712293752|
|    min|               16|
|    max|               36|
+-------+-----------------+
```

##### Part 3: Data visualization

```python
fifa_df_germany_age_pandas = fifa_df_germany_age.toPandas()

# Plot the 'Age' density of Germany Players
fifa_df_germany_age_pandas.plot(kind='density')
plt.show()

>>>
...
```



<br>

## Clearning Data with PySpark

### 1. DataFrame details

#### Defining a schema

```python
# Import the pyspark.sql.types library
from pyspark.sql.types import *

# Define a new schema using the StructType method
people_schema = StructType([
  # Define a StructField for each field
  StructField('name', StringType(), False), # False: not be nullable
  StructField('age', IntegerType(), False),
  StructField('city', StringType(), True)
])
```

Immutability 

* A component of functional programming
* Defined once
* Unable to be directly modified
* Re-created if reassigned
* Able to be shared efficiently

Lazy Processing

* Transformations
* Actions
* Allows efficient planning

#### Using lazy processing

Lazy processing operations will usually return in about the same amount of time regardless of the actual quantity of data. Remember that this is due to Spark not performing any transformations until an action is requested.

```python
# Load the CSV file
aa_dfw_df = spark.read.format('csv').options(Header=True).load('AA_DFW_2018.csv.gz')

# Add the airport column using the F.lower() method
aa_dfw_df = aa_dfw_df.withColumn('airport', F.lower(aa_dfw_df['Destination Airport']))

# Drop the Destination Airport column
aa_dfw_df = aa_dfw_df.drop(aa_dfw_df['Destination Airport'])

# Show the DataFrame
aa_dfw_df.show()
```

```
>>>
+-----------------+-------------+-----------------------------+-------+
|Date (MM/DD/YYYY)|Flight Number|Actual elapsed time (Minutes)|airport|
+-----------------+-------------+-----------------------------+-------+
|       01/01/2018|         0005|                          498|    hnl|
|       01/01/2018|         0007|                          501|    ogg|
|       01/01/2018|         0043|                            0|    dtw|
|       01/01/2018|         0051|                          100|    stl|
|       01/01/2018|         0075|                          147|    dca|
|       01/01/2018|         0096|                           92|    stl|
|       01/01/2018|         0103|                          227|    sjc|
|       01/01/2018|         0119|                          517|    ogg|
|       01/01/2018|         0123|                          489|    hnl|
|       01/01/2018|         0128|                          141|    mco|
|       01/01/2018|         0132|                          201|    ewr|
|       01/01/2018|         0140|                          215|    sjc|
|       01/01/2018|         0174|                          140|    rdu|
|       01/01/2018|         0190|                           68|    sat|
|       01/01/2018|         0200|                          215|    sfo|
|       01/01/2018|         0209|                          169|    mia|
|       01/01/2018|         0217|                          178|    las|
|       01/01/2018|         0229|                          534|    koa|
|       01/01/2018|         0244|                          115|    cvg|
|       01/01/2018|         0262|                          159|    mia|
+-----------------+-------------+-----------------------------+-------+
only showing top 20 rows
```

Parquet Format 

* A columnar data format
* Supported in Spark and other data processing framewords
* Supports predicate pushdown
* Automatically stores schema information

#### Saving a DataFrame in Parquet format

When working with Spark, you'll often start with CSV, JSON, or other data sources. This provides a lot of flexibility for the types of data to load, but it is not an optimal format for Spark. The `Parquet` format is a columnar data store, allowing Spark to use predicate pushdown. This means Spark will only process the data necessary to complete the operations you define versus reading the entire dataset. This gives Spark more flexibility in accessing the data and often drastically improves performance on large datasets.

```python
# View the row count of df1 and df2
print("df1 Count: %d" % df1.count())
print("df2 Count: %d" % df2.count())

# Combine the DataFrames into one 
df3 = df1.union(df2)

# Save the df3 DataFrame in Parquet format
df3.write.parquet('AA_DFW_ALL.parquet', mode='overwrite')

# Read the Parquet file into a new DataFrame and run a count
print(spark.read.parquet('AA_DFW_ALL.parquet').count())
```
```
>>>
df1 Count: 139359
df2 Count: 119911
259270
```

#### SQL and Parquet

Parquet files are perfect as a backing data store for SQL queries in Spark. While it is possible to run the same queries directly via Spark's Python functions, sometimes it's easier to run SQL queries alongside the Python options.

```python
# Read the Parquet file into flights_df
flights_df = spark.read.parquet('AA_DFW_ALL.parquet')

# Register the temp table
flights_df.createOrReplaceTempView('flights')

# Run a SQL query of the average flight duration
avg_duration = spark.sql('SELECT avg(flight_duration) from flights').collect()[0]
print('The average flight time is: %d' % avg_duration)
```
```
>>>
The average flight time is: 151
```

### 2. Manipulating DataFrames in the real world

#### Filtering column content with Python

```python
# Show the distinct VOTER_NAME entries
voter_df.select(voter_df['VOTER_NAME']).distinct().show(10, truncate=False)

# Filter voter_df where the VOTER_NAME is 1-20 characters in length
voter_df = voter_df.filter('length(VOTER_NAME) > 0 and length(VOTER_NAME) < 20')

# Filter out voter_df where the VOTER_NAME contains an underscore
voter_df = voter_df.filter(~ F.col('VOTER_NAME').contains('_'))

# Show the distinct VOTER_NAME entries again
voter_df.select('VOTER_NAME').distinct().show(10, truncate=False)
```
```
>>>
+-------------------+
|VOTER_NAME         |
+-------------------+
|Tennell Atkins     |
|Scott Griggs       |
|Scott  Griggs      |
|Sandy Greyson      |
|Michael S. Rawlings|
|Kevin Felder       |
|Adam Medrano       |
|Casey  Thomas      |
|Mark  Clayton      |
|Casey Thomas       |
+-------------------+
only showing top 10 rows

+-------------------+
|VOTER_NAME         |
+-------------------+
|Tennell Atkins     |
|Scott Griggs       |
|Scott  Griggs      |
|Sandy Greyson      |
|Michael S. Rawlings|
|Kevin Felder       |
|Adam Medrano       |
|Casey  Thomas      |
|Mark  Clayton      |
|Casey Thomas       |
+-------------------+
only showing top 10 rows
```

#### Modifying DataFrame columns

```python
# Add a new column called splits separated on whitespace
voter_df = voter_df.withColumn('splits', F.split(voter_df.VOTER_NAME, '\s+'))

# Create a new column called first_name based on the first item in splits
voter_df = voter_df.withColumn('first_name', voter_df.splits.getItem(0))

# Get the last entry of the splits list and create a column called last_name
voter_df = voter_df.withColumn('last_name', voter_df.splits.getItem(F.size('splits') - 1))

# Drop the splits column
#voter_df = voter_df.drop('splits')

# Show the voter_df DataFrame
voter_df.show()
```
```
>>>
+----------+-------------+-------------------+----------+---------+--------------------+
|      DATE|        TITLE|         VOTER_NAME|first_name|last_name|              splits|
+----------+-------------+-------------------+----------+---------+--------------------+
|02/08/2017|Councilmember|  Jennifer S. Gates|  Jennifer|    Gates|[Jennifer, S., Ga...|
|02/08/2017|Councilmember| Philip T. Kingston|    Philip| Kingston|[Philip, T., King...|
|02/08/2017|        Mayor|Michael S. Rawlings|   Michael| Rawlings|[Michael, S., Raw...|
|02/08/2017|Councilmember|       Adam Medrano|      Adam|  Medrano|     [Adam, Medrano]|
|02/08/2017|Councilmember|       Casey Thomas|     Casey|   Thomas|     [Casey, Thomas]|
|02/08/2017|Councilmember|Carolyn King Arnold|   Carolyn|   Arnold|[Carolyn, King, A...|
|02/08/2017|Councilmember|       Scott Griggs|     Scott|   Griggs|     [Scott, Griggs]|
|02/08/2017|Councilmember|   B. Adam  McGough|        B.|  McGough| [B., Adam, McGough]|
|02/08/2017|Councilmember|       Lee Kleinman|       Lee| Kleinman|     [Lee, Kleinman]|
|02/08/2017|Councilmember|      Sandy Greyson|     Sandy|  Greyson|    [Sandy, Greyson]|
|02/08/2017|Councilmember|  Jennifer S. Gates|  Jennifer|    Gates|[Jennifer, S., Ga...|
|02/08/2017|Councilmember| Philip T. Kingston|    Philip| Kingston|[Philip, T., King...|
|02/08/2017|        Mayor|Michael S. Rawlings|   Michael| Rawlings|[Michael, S., Raw...|
|02/08/2017|Councilmember|       Adam Medrano|      Adam|  Medrano|     [Adam, Medrano]|
|02/08/2017|Councilmember|       Casey Thomas|     Casey|   Thomas|     [Casey, Thomas]|
|02/08/2017|Councilmember|Carolyn King Arnold|   Carolyn|   Arnold|[Carolyn, King, A...|
|02/08/2017|Councilmember| Rickey D. Callahan|    Rickey| Callahan|[Rickey, D., Call...|
|01/11/2017|Councilmember|  Jennifer S. Gates|  Jennifer|    Gates|[Jennifer, S., Ga...|
|04/25/2018|Councilmember|     Sandy  Greyson|     Sandy|  Greyson|    [Sandy, Greyson]|
|04/25/2018|Councilmember| Jennifer S.  Gates|  Jennifer|    Gates|[Jennifer, S., Ga...|
+----------+-------------+-------------------+----------+---------+--------------------+
only showing top 20 rows
```

#### when() example

```python
# Add a column to voter_df for any voter with the title **Councilmember**
voter_df = voter_df.withColumn('random_val', 
							   when(voter_df.TITLE == 'Councilmember', F.rand()))

# Show some of the DataFrame rows, noting whether the when clause worked
voter_df.show()
```
```
>>>
+----------+-------------+-------------------+-------------------+
|      DATE|        TITLE|         VOTER_NAME|         random_val|
+----------+-------------+-------------------+-------------------+
|02/08/2017|Councilmember|  Jennifer S. Gates| 0.5259552691048351|
|02/08/2017|Councilmember| Philip T. Kingston|0.17625589832860167|
|02/08/2017|        Mayor|Michael S. Rawlings|               null|
|02/08/2017|Councilmember|       Adam Medrano|0.18619643510090478|
|02/08/2017|Councilmember|       Casey Thomas| 0.5443793016444369|
|02/08/2017|Councilmember|Carolyn King Arnold|0.21635815801488967|
|02/08/2017|Councilmember|       Scott Griggs| 0.3462673569610931|
|02/08/2017|Councilmember|   B. Adam  McGough|0.13248863658190047|
|02/08/2017|Councilmember|       Lee Kleinman| 0.9988060287273388|
|02/08/2017|Councilmember|      Sandy Greyson|0.21568269860777767|
|02/08/2017|Councilmember|  Jennifer S. Gates|0.38725871611028617|
|02/08/2017|Councilmember| Philip T. Kingston|0.30660836268346003|
|02/08/2017|        Mayor|Michael S. Rawlings|               null|
|02/08/2017|Councilmember|       Adam Medrano|0.29597144654635144|
|02/08/2017|Councilmember|       Casey Thomas| 0.2084740566202885|
|02/08/2017|Councilmember|Carolyn King Arnold| 0.3471425496068026|
|02/08/2017|Councilmember| Rickey D. Callahan|  0.918118430581971|
|01/11/2017|Councilmember|  Jennifer S. Gates|0.27398290238813605|
|04/25/2018|Councilmember|     Sandy  Greyson| 0.1277703989854202|
|04/25/2018|Councilmember| Jennifer S.  Gates| 0.8508697003533658|
+----------+-------------+-------------------+-------------------+
only showing top 20 rows
```

#### when / otherwise

```python
# Add a column to voter_df for a voter based on their position
voter_df = voter_df.withColumn('random_val',
                               when(voter_df.TITLE == 'Councilmember', F.rand())
                               .when(voter_df.TITLE == 'Mayor', 2)
                               .otherwise(0))

# Show some of the DataFrame rows
voter_df.show()

# Use the .filter() clause with random_val
voter_df.filter(voter_df.random_val == 0).show()
```
```
>>>
+----------+-------------+-------------------+-------------------+
|      DATE|        TITLE|         VOTER_NAME|         random_val|
+----------+-------------+-------------------+-------------------+
|02/08/2017|Councilmember|  Jennifer S. Gates| 0.5335180413446635|
|02/08/2017|Councilmember| Philip T. Kingston| 0.5239942280961327|
|02/08/2017|        Mayor|Michael S. Rawlings|                2.0|
|02/08/2017|Councilmember|       Adam Medrano| 0.9553668855576738|
|02/08/2017|Councilmember|       Casey Thomas| 0.5395390808757042|
|02/08/2017|Councilmember|Carolyn King Arnold|0.09125153438186318|
|02/08/2017|Councilmember|       Scott Griggs|0.07872421499768645|
|02/08/2017|Councilmember|   B. Adam  McGough| 0.0392908798310857|
|02/08/2017|Councilmember|       Lee Kleinman| 0.4253681842997483|
|02/08/2017|Councilmember|      Sandy Greyson| 0.4547810469927127|
|02/08/2017|Councilmember|  Jennifer S. Gates|0.47813902838457556|
|02/08/2017|Councilmember| Philip T. Kingston| 0.3230747590250692|
|02/08/2017|        Mayor|Michael S. Rawlings|                2.0|
|02/08/2017|Councilmember|       Adam Medrano|0.14982267724140197|
|02/08/2017|Councilmember|       Casey Thomas| 0.3339422468104286|
|02/08/2017|Councilmember|Carolyn King Arnold|0.05516094068187882|
|02/08/2017|Councilmember| Rickey D. Callahan|  0.922133072469369|
|01/11/2017|Councilmember|  Jennifer S. Gates| 0.8154216508949929|
|04/25/2018|Councilmember|     Sandy  Greyson|0.15876843281762565|
|04/25/2018|Councilmember| Jennifer S.  Gates| 0.1569369988866156|
+----------+-------------+-------------------+-------------------+
only showing top 20 rows

+----------+--------------------+-----------------+----------+
|      DATE|               TITLE|       VOTER_NAME|random_val|
+----------+--------------------+-----------------+----------+
|04/25/2018|Deputy Mayor Pro Tem|     Adam Medrano|       0.0|
|04/25/2018|       Mayor Pro Tem|Dwaine R. Caraway|       0.0|
|06/20/2018|Deputy Mayor Pro Tem|     Adam Medrano|       0.0|
|06/20/2018|       Mayor Pro Tem|Dwaine R. Caraway|       0.0|
|06/20/2018|Deputy Mayor Pro Tem|     Adam Medrano|       0.0|
|06/20/2018|       Mayor Pro Tem|Dwaine R. Caraway|       0.0|
|08/15/2018|Deputy Mayor Pro Tem|     Adam Medrano|       0.0|
|08/15/2018|Deputy Mayor Pro Tem|     Adam Medrano|       0.0|
|09/18/2018|Deputy Mayor Pro Tem|     Adam Medrano|       0.0|
|09/18/2018|       Mayor Pro Tem|    Casey  Thomas|       0.0|
|04/25/2018|Deputy Mayor Pro Tem|     Adam Medrano|       0.0|
|04/25/2018|       Mayor Pro Tem|Dwaine R. Caraway|       0.0|
|04/11/2018|       Mayor Pro Tem|Dwaine R. Caraway|       0.0|
|04/11/2018|Deputy Mayor Pro Tem|     Adam Medrano|       0.0|
|04/11/2018|       Mayor Pro Tem|Dwaine R. Caraway|       0.0|
|04/11/2018|Deputy Mayor Pro Tem|     Adam Medrano|       0.0|
|04/11/2018|       Mayor Pro Tem|Dwaine R. Caraway|       0.0|
|06/13/2018|Deputy Mayor Pro Tem|     Adam Medrano|       0.0|
|06/13/2018|       Mayor Pro Tem|Dwaine R. Caraway|       0.0|
|04/11/2018|Deputy Mayor Pro Tem|     Adam Medrano|       0.0|
+----------+--------------------+-----------------+----------+
only showing top 20 rows
```

#### Using user defined functions in Spark

```python
def getFirstAndMiddle(names):
  # Return a space separated string of names
  return ' '.join(names[:-1])

# Define the method as a UDF
udfFirstAndMiddle = F.udf(getFirstAndMiddle, StringType())

# Create a new column using your UDF
voter_df = voter_df.withColumn('first_and_middle_name', udfFirstAndMiddle(voter_df.splits))

# Drop the unnecessary columns then show the DataFrame
voter_df = voter_df.drop('first_name')
voter_df = voter_df.drop('splits')
voter_df.show()
```
```
>>>
+----------+-------------+-------------------+---------+---------------------+
|      DATE|        TITLE|         VOTER_NAME|last_name|first_and_middle_name|
+----------+-------------+-------------------+---------+---------------------+
|02/08/2017|Councilmember|  Jennifer S. Gates|    Gates|          Jennifer S.|
|02/08/2017|Councilmember| Philip T. Kingston| Kingston|            Philip T.|
|02/08/2017|        Mayor|Michael S. Rawlings| Rawlings|           Michael S.|
|02/08/2017|Councilmember|       Adam Medrano|  Medrano|                 Adam|
|02/08/2017|Councilmember|       Casey Thomas|   Thomas|                Casey|
|02/08/2017|Councilmember|Carolyn King Arnold|   Arnold|         Carolyn King|
|02/08/2017|Councilmember|       Scott Griggs|   Griggs|                Scott|
|02/08/2017|Councilmember|   B. Adam  McGough|  McGough|              B. Adam|
|02/08/2017|Councilmember|       Lee Kleinman| Kleinman|                  Lee|
|02/08/2017|Councilmember|      Sandy Greyson|  Greyson|                Sandy|
|02/08/2017|Councilmember|  Jennifer S. Gates|    Gates|          Jennifer S.|
|02/08/2017|Councilmember| Philip T. Kingston| Kingston|            Philip T.|
|02/08/2017|        Mayor|Michael S. Rawlings| Rawlings|           Michael S.|
|02/08/2017|Councilmember|       Adam Medrano|  Medrano|                 Adam|
|02/08/2017|Councilmember|       Casey Thomas|   Thomas|                Casey|
|02/08/2017|Councilmember|Carolyn King Arnold|   Arnold|         Carolyn King|
|02/08/2017|Councilmember| Rickey D. Callahan| Callahan|            Rickey D.|
|01/11/2017|Councilmember|  Jennifer S. Gates|    Gates|          Jennifer S.|
|04/25/2018|Councilmember|     Sandy  Greyson|  Greyson|                Sandy|
|04/25/2018|Councilmember| Jennifer S.  Gates|    Gates|          Jennifer S.|
+----------+-------------+-------------------+---------+---------------------+
only showing top 20 rows
```

Partitioning

* DataFrames are broken up into partitions
* Partition size can vary
* Each partition is handled independently

Lazy processing

* Transformations are lazy
   * .withColumn()
   * .select()
* Nothing is actually done until an action is performed
   * .count()
   * .write()
* Transformations can be re-ordered for best performance
* Sometimes causes unexpected behavior

Adding IDs

* Normal ID fields
   * Common in relational databases
   * Most usually an integer increasing, sequential and unique
   * Not very parallel (단일 서버는 괜찮지만, 분산 서버와 같은 Spark에서는 문제점)

Monotonically increasing IDs

* pyspark.sql.functions.monotonically_increasing_id()
   * Integer(64-bit), increases in value, unique
   * Not necessarily sequential (gaps exist)
   * Completely parallel

Spark is lazy!

* Occasionally out of order
* If performing a join, ID may be assigned after the join
* Test your transformations

#### Adding an ID Field

When working with data, you sometimes only want to access certain fields and perform various operations. In this case, find all the unique voter names from the DataFrame and add a unique ID number. Remember that Spark IDs are assigned based on the DataFrame partition - as such the ID values may be much greater than the actual number of rows in the DataFrame.

With Spark's lazy processing, the IDs are not actually generated until an action is performed and can be somewhat random depending on the size of the dataset.

```python
# The pyspark.sql.functions library is available under the alias F.

# Select all the unique council voters
voter_df = df.select(df["VOTER NAME"]).distinct()

# Count the rows in voter_df
print("\nThere are %d rows in the voter_df DataFrame.\n" % voter_df.count())

# Add a ROW_ID
voter_df = voter_df.withColumn('ROW_ID', F.monotonically_increasing_id())

# Show the rows with 10 highest IDs in the set
voter_df.orderBy(voter_df.ROW_ID.desc()).show(10)
```
```
>>>
There are 36 rows in the voter_df DataFrame.

+--------------------+-------------+
|          VOTER NAME|       ROW_ID|
+--------------------+-------------+
|        Lee Kleinman|1709396983808|
|  the  final  201...|1700807049217|
|         Erik Wilson|1700807049216|
|  the  final   20...|1683627180032|
| Carolyn King Arnold|1632087572480|
| Rickey D.  Callahan|1597727834112|
|   the   final  2...|1443109011456|
|    Monica R. Alonzo|1382979469312|
|     Lee M. Kleinman|1228360646656|
|   Jennifer S. Gates|1194000908288|
+--------------------+-------------+
only showing top 10 rows
```

#### IDs with different partitions

```python
# Print the number of partitions in each DataFrame
print("\nThere are %d partitions in the voter_df DataFrame.\n" % voter_df.rdd.getNumPartitions())
print("\nThere are %d partitions in the voter_df_single DataFrame.\n" % voter_df_single.rdd.getNumPartitions())

# Add a ROW_ID field to each DataFrame
voter_df = voter_df.withColumn('ROW_ID', F.monotonically_increasing_id())
voter_df_single = voter_df_single.withColumn('ROW_ID', F.monotonically_increasing_id())

# Show the top 10 IDs in each DataFrame 
voter_df.orderBy(voter_df.ROW_ID.desc()).show(10)
voter_df_single.orderBy(voter_df_single.ROW_ID.desc()).show(10)
```
```
>>>
There are 200 partitions in the voter_df DataFrame.


There are 1 partitions in the voter_df_single DataFrame.

+--------------------+-------------+
|          VOTER NAME|       ROW_ID|
+--------------------+-------------+
|        Lee Kleinman|1709396983808|
|  the  final  201...|1700807049217|
|         Erik Wilson|1700807049216|
|  the  final   20...|1683627180032|
| Carolyn King Arnold|1632087572480|
| Rickey D.  Callahan|1597727834112|
|   the   final  2...|1443109011456|
|    Monica R. Alonzo|1382979469312|
|     Lee M. Kleinman|1228360646656|
|   Jennifer S. Gates|1194000908288|
+--------------------+-------------+
only showing top 10 rows

+--------------------+------+
|          VOTER NAME|ROW_ID|
+--------------------+------+
|        Lee Kleinman|    35|
|  the  final  201...|    34|
|         Erik Wilson|    33|
|  the  final   20...|    32|
| Carolyn King Arnold|    31|
| Rickey D.  Callahan|    30|
|   the   final  2...|    29|
|    Monica R. Alonzo|    28|
|     Lee M. Kleinman|    27|
|   Jennifer S. Gates|    26|
+--------------------+------+
only showing top 10 rows
```

#### More ID tricks

Once you define a Spark process, you'll likely want to use it many times. Depending on your needs, you may want to start your IDs at a certain value so there isn't overlap with previous runs of the Spark task. This behavior is similar to how IDs would behave in a relational database. You have been given the task to make sure that the IDs output from a monthly Spark task start at the highest value from the previous month.

```python
# Determine the highest ROW_ID and save it in previous_max_ID
previous_max_ID = voter_df_march.select('ROW_ID').rdd.max()[0]

# Add a ROW_ID column to voter_df_april starting at the desired value
voter_df_april = voter_df_april.withColumn('ROW_ID', F.monotonically_increasing_id() + previous_max_ID)

# Show the ROW_ID from both DataFrames and compare
voter_df_march.select('ROW_ID').show()
voter_df_april.select('ROW_ID').show()
```
```
>>>
+-------------+
|       ROW_ID|
+-------------+
|   8589934592|
|  25769803776|
|  34359738368|
|  42949672960|
|  51539607552|
| 103079215104|
| 111669149696|
| 231928233984|
| 240518168576|
| 360777252864|
| 395136991232|
| 601295421440|
| 635655159808|
| 670014898176|
| 807453851648|
| 850403524608|
| 944892805120|
| 962072674304|
|1005022347264|
|1047972020224|
+-------------+
only showing top 20 rows

+-------------+
|       ROW_ID|
+-------------+
|1717986918400|
|1735166787584|
|1743756722176|
|1752346656768|
|1760936591360|
|1812476198912|
|1821066133504|
|1941325217792|
|1949915152384|
|2070174236672|
|2104533975040|
|2310692405248|
|2345052143616|
|2379411881984|
|2516850835456|
|2559800508416|
|2654289788928|
|2671469658112|
|2714419331072|
|2757369004032|
+-------------+
only showing top 20 rows
```

### 3. Improving Performance

Caching in Spark

* Stores DataFrames in memory or on disk
* Improves speed on later transformations / actions
* Reduces resource usage

Disadvantages of caching

* Very large data sets may not fit in memory
* Local disk based caching may not be a performance improvement
* Cached objects may not be available

Cache Tips

* Cache only if you need it
* Try caching DataFrames at various points and determine if your performance improves
* Cache in memory and fast SSD / NVMe storage
* Cache to slow local disk if needed
* Use intermediate files!
* Stop caching objects when finished

#### Caching a DataFrame

You've been assigned a task that requires running several analysis operations on a DataFrame. You've learned that caching can improve performance when reusing DataFrames and would like to implement it.

```python
start_time = time.time()

# Add caching to the unique rows in departures_df
departures_df = departures_df.distinct().cache()

# Count the unique rows in departures_df, noting how long the operation takes
print("Counting %d rows took %f seconds" % (departures_df.count(), time.time() - start_time))

# Count the rows again, noting the variance in time of a cached DataFrame
start_time = time.time()
print("Counting %d rows again took %f seconds" % (departures_df.count(), time.time() - start_time))
```
```
Counting 139358 rows took 3.701810 seconds
Counting 139358 rows again took 0.778054 seconds
```

#### Removing a DataFrame from cache

You've finished the analysis tasks with the departures_df DataFrame, but have some other processing to do. You'd like to remove the DataFrame from the cache to prevent any excess memory usage on your cluster.

```python
# Determine if departures_df is in the cache
print("Is departures_df cached?: %s" % departures_df.is_cached)
print("Removing departures_df from cache")

# Remove departures_df from the cache
departures_df.unpersist()

# Check the cache status again
print("Is departures_df cached?: %s" % departures_df.is_cached)
```
```
Is departures_df cached?: True
Removing departures_df from cache
Is departures_df cached?: False
```

Spark clusters are made of two types of processes

* Driver process
* Worker process

Import parameters

* Number of objects (Files, Network locations, etc)
   * More objects better than larger ones
   * Can import via wildcard `airport_df = spark.read.csv('airports-*.txt.gz)`
* General size of objects
   * Spark performs better if objects are of similar size

A well-defined schema will drastically improve import performance

* Avoids reading the data multiple times
* Provides validation on import

How to split objects

* Use OS utilities / scripts (split, cut, awk) `split -l 10000 -d largefile chunk-`
* Use custom scripts (python)
* Write out to Parquet

File size optimization

* (현재 상황) 2 Large data files on a cluster with 10 nodes. Each file contains 10M rows of roughly the same size.
* (문제) the responsiveness is acceptable but the inital read from the files takes a considerable period time.
* (해결) split the 2 files into 50 files of 400K rows each.

#### File import performance

You have two types of files available: `departures_full.txt.gz` and `departures_xxx.txt.gz` where xxx is 000 - 013. The same number of rows is split between each file.

```python
# Import the full and split files into DataFrames
full_df = spark.read.csv('departures_full.txt.gz')
split_df = spark.read.csv('departures_0*.txt.gz')

# Print the count and run time for each DataFrame
start_time_a = time.time()
print("Total rows in full DataFrame:\t%d" % full_df.count())
print("Time to run: %f" % (time.time() - start_time_a))

start_time_b = time.time()
print("Total rows in split DataFrame:\t%d" % split_df.count())
print("Time to run: %f" % (time.time() - start_time_b))
```
```
>>>
Total rows in full DataFrame:	139359
Time to run: 0.925659
Total rows in split DataFrame:	139359
Time to run: 0.306183
```

Spark deployment options:
* Single node
* Standalone
* Managed
   * YARN
   * Mesos
   * Kubernetes

Driver
* Task assignment
* Result consolidation
* Shared data access

Driver Tips

* Driver node should double the memory of the worker (useful for task monitoring, consolidation task)
* Fast local storage helpful

Worker

* Runs actual tasks
* Ideally has all code, data, and resources for a given task

Worker Recommendations

* More worker nodes is often better than larger workers
* Test to find the balance
* Fast local storage extremely useful

#### Reading Spark configurations

```python
# Name of the Spark application instance
app_name = spark.conf.get('spark.app.name')

# Driver TCP port
driver_tcp_port = spark.conf.get('spark.driver.port')

# Number of join partitions
num_partitions = spark.conf.get('spark.sql.shuffle.partitions')

# Show the results
print("Name: %s" % app_name)
print("Driver TCP port: %s" % driver_tcp_port)
print("Number of partitions: %s" % num_partitions)
```
```
>>>
Name: pyspark-shell
Driver TCP port: 41447
Number of partitions: 200
```

#### Writing Spark configurations

```python
# Store the number of partitions in variable
before = departures_df.rdd.getNumPartitions()

# Configure Spark to use 500 partitions
spark.conf.set('spark.sql.shuffle.partitions', 500)

# Recreate the DataFrame using the departures data file
departures_df = spark.read.csv('departures.txt.gz').distinct()

# Print the number of partitions for each instance
print("Partition count before change: %d" % before)
print("Partition count after change: %d" % departures_df.rdd.getNumPartitions())
```
```
>>>
Partition count before change: 200
Partition count after change: 500
```

Explaining the Spark execution plan
* `.explain()`

Suffling: moving data around to various workers to complete task
* Hides complexity from the user (don't need to know which node has what data)
* Can be slow to complete
* Lowers overall throughput
* Is often necessary, but try to minimize

How to limit shuffling?
* Limit use of `.repartition(num_partitions)`
   * Use `.coalesce(num_partitions)` instead
* Be careful when calling `.join()`
* Use `.broadcast()`
* May not need to limit it 

Broadcasting

* Provides a copy of an object to each worker
* Prevents undue / excess communication between nodes
* Can drastically speed up `.join()` operations
* 너무 작거나 또는 너무 큰 데이터를 broadcasting해서 join하면 느려질수도 있음 (내부적으로 최적화하긴 하지만 조심)

#### Normal joins

```python
# Join the flights_df and aiports_df DataFrames
normal_df = flights_df.join(airports_df, \
    flights_df["Destination Airport"] == airports_df["IATA"] )

# Show the query plan
normal_df.explain()
```
```
>>>
== Physical Plan ==
*(5) SortMergeJoin [Destination Airport#249], [IATA#266], Inner
:- *(2) Sort [Destination Airport#249 ASC NULLS FIRST], false, 0
:  +- Exchange hashpartitioning(Destination Airport#249, 500)
:     +- *(1) Project [Date (MM/DD/YYYY)#247, Flight Number#248, Destination Airport#249, Actual elapsed time (Minutes)#250]
:        +- *(1) Filter isnotnull(Destination Airport#249)
:           +- *(1) FileScan csv [Date (MM/DD/YYYY)#247,Flight Number#248,Destination Airport#249,Actual elapsed time (Minutes)#250] Batched: false, Format: CSV, Location: InMemoryFileIndex[file:/tmp/tmps550ugm8/AA_DFW_2018_Departures_Short.csv.gz], PartitionFilters: [], PushedFilters: [IsNotNull(Destination Airport)], ReadSchema: struct<Date (MM/DD/YYYY):string,Flight Number:string,Destination Airport:string,Actual elapsed ti...
+- *(4) Sort [IATA#266 ASC NULLS FIRST], false, 0
   +- Exchange hashpartitioning(IATA#266, 500)
      +- *(3) Project [AIRPORTNAME#265, IATA#266]
         +- *(3) Filter isnotnull(IATA#266)
            +- *(3) FileScan csv [AIRPORTNAME#265,IATA#266] Batched: false, Format: CSV, Location: InMemoryFileIndex[file:/tmp/tmps550ugm8/airportnames.txt.gz], PartitionFilters: [], PushedFilters: [IsNotNull(IATA)], ReadSchema: struct<AIRPORTNAME:string,IATA:string>
```

#### Using broadcasting on Spark joins

Remember that table joins in Spark are split between the cluster workers. If the data is not local, various shuffle operations are required and can have a negative impact on performance. Instead, we're going to use Spark's broadcast operations to give each node a copy of the specified data.

A couple tips:
* Broadcast the smaller DataFrame. The larger the DataFrame, the more time required to transfer to the worker nodes.
* On small DataFrames, it may be better skip broadcasting and let Spark figure out any optimization on its own.
* If you look at the query execution plan, a broadcastHashJoin indicates you've successfully configured broadcasting.

```python
# Import the broadcast method from pyspark.sql.functions
from pyspark.sql.functions import broadcast

# Join the flights_df and airports_df DataFrames using broadcasting
broadcast_df = flights_df.join(broadcast(airports_df), \
    flights_df["Destination Airport"] == airports_df["IATA"] )

# Show the query plan and compare against the original
broadcast_df.explain()
```
```
>>>
== Physical Plan ==
*(2) BroadcastHashJoin [Destination Airport#335], [IATA#352], Inner, BuildRight
:- *(2) Project [Date (MM/DD/YYYY)#333, Flight Number#334, Destination Airport#335, Actual elapsed time (Minutes)#336]
:  +- *(2) Filter isnotnull(Destination Airport#335)
:     +- *(2) FileScan csv [Date (MM/DD/YYYY)#333,Flight Number#334,Destination Airport#335,Actual elapsed time (Minutes)#336] Batched: false, Format: CSV, Location: InMemoryFileIndex[file:/tmp/tmps550ugm8/AA_DFW_2018_Departures_Short.csv.gz], PartitionFilters: [], PushedFilters: [IsNotNull(Destination Airport)], ReadSchema: struct<Date (MM/DD/YYYY):string,Flight Number:string,Destination Airport:string,Actual elapsed ti...
+- BroadcastExchange HashedRelationBroadcastMode(List(input[1, string, true]))
   +- *(1) Project [AIRPORTNAME#351, IATA#352]
      +- *(1) Filter isnotnull(IATA#352)
         +- *(1) FileScan csv [AIRPORTNAME#351,IATA#352] Batched: false, Format: CSV, Location: InMemoryFileIndex[file:/tmp/tmps550ugm8/airportnames.txt.gz], PartitionFilters: [], PushedFilters: [IsNotNull(IATA)], ReadSchema: struct<AIRPORTNAME:string,IATA:string>
```

#### Comparing broadcase vs. normal joins

```python
start_time = time.time()
# Count the number of rows in the normal DataFrame
normal_count = normal_df.count()
normal_duration = time.time() - start_time

start_time = time.time()
# Count the number of rows in the broadcast DataFrame
broadcast_count = broadcast_df.count()
broadcast_duration = time.time() - start_time

# Print the counts and the duration of the tests
print("Normal count:\t\t%d\tduration: %f" % (normal_count, normal_duration))
print("Broadcast count:\t%d\tduration: %f" % (broadcast_count, broadcast_duration))
```
```
>>>
Normal count:		119910	duration: 4.781047
Broadcast count:	119910	duration: 0.811799
```

### 4. Complex processing and data pipelines






<br>


## Building Recommendation Engines with PySpark

### 1. Recommendations Are Everywhere

일반적인 파라미터 최적화 기법은 Gradient Descent가 있다. 분산처리 환경에서는 GD보다는 ALS(Alternating Least Squares)가 더 효과적이라고 알려져 있다.

#### See the power of a recommendation engine

```python
# View TJ_ratings
TJ_ratings.show()

# Generate recommendations for users
get_ALS_recs(["Jane", "Taylor"])
```
```
+---------+--------------------+------+
|user_name|          movie_name|rating|
+---------+--------------------+------+
|   Taylor|            Twilight|   4.9|
|   Taylor|  A Walk to Remember|   4.5|
|   Taylor|        The Notebook|   5.0|
|   Taylor|Raiders of the Lo...|   1.2|
|   Taylor|      The Terminator|   1.0|
|   Taylor|      Mrs. Doubtfire|   1.0|
|     Jane|            Iron Man|   4.8|
|     Jane|Raiders of the Lo...|   4.9|
|     Jane|      The Terminator|   4.6|
|     Jane|           Anchorman|   1.2|
|     Jane|        Pretty Woman|   1.0|
|     Jane|           Toy Story|   1.2|
+---------+--------------------+------+

    userId  pred_rating                 title          genres
0   Taylor         3.89   Seven Pounds (2008)           Drama
1   Taylor         3.61      Cure, The (1995)           Drama
2   Taylor         3.55  Kiss Me, Guido (1997          Comedy
3   Taylor         3.29  You've Got Mail (199  Comedy|Romance
4   Taylor         3.27  10 Things I Hate Abo  Comedy|Romance
5   Taylor         3.26  Corrina, Corrina (19  Comedy|Drama|R
6     Jane         4.96           Fear (1996)        Thriller
7     Jane         4.85  Lord of the Rings: T  Adventure|Fant
8     Jane         4.70  Lord of the Rings: T  Adventure|Fant
9     Jane         4.55  No Holds Barred (198          Action
10    Jane         4.54  Lord of the Rings: T  Action|Adventu
11    Jane         4.30  Band of Brothers (20  Action|Drama|W
12    Jane         4.26   Transformers (2007)  Action|Sci-Fi|
```

#### Collaborative vs Content-Based Filtering

아래의 df를 사용해서 Both collaborative and content-based filtering 를 할 수 있음.


```python
df.show()
```
```
+------+-------+-----------------+--------+--------+-------------+------+
|UserId|MovieId|      Movie_Title|   Genre|Language|Year_Produced|rating|
+------+-------+-----------------+--------+--------+-------------+------+
| User1|   2112|     Finding Nemo|Animated| English|         2003|     3|
| User1|   2113|   The Terminator|  Action| English|         1984|     0|
| User1|   2114|       Spinal Tap|  Satire| English|         1984|     4|
| User1|   2115|Life Is Beautiful|   Drama| Italian|         1998|     4|
| User2|   2112|     Finding Nemo|Animated| English|         2003|     4|
| User2|   2113|   The Terminator|  Action| English|         1984|     0|
| User2|   2114|       Spinal Tap|  Satire| English|         1984|     0|
| User2|   2115|Life Is Beautiful|   Drama| Italian|         1998|     4|
| User3|   2112|     Finding Nemo|Animated| English|         2003|     1|
| User3|   2113|   The Terminator|  Action| English|         1984|     2|
| User3|   2114|       Spinal Tap|  Satire| English|         1984|     1|
| User3|   2115|Life Is Beautiful|   Drama| Italian|         1998|     0|
| User4|   2112|     Finding Nemo|Animated| English|         2003|     3|
| User4|   2113|   The Terminator|  Action| English|         1984|     1|
| User4|   2114|       Spinal Tap|  Satire| English|         1984|     0|
| User4|   2115|Life Is Beautiful|   Drama| Italian|         1998|     0|
+------+-------+-----------------+--------+--------+-------------+------+
```

#### Implicit vs Explicit Data

`Implicit` Data 의 모습은 다음과 같다. Explici와 달리 아이템에 대한 사용자의 호불호를 정확하게 파악할 수 없다. 예를 들어, 상품을 구매했다고 반드시 그 상품에 호의적인 평가를 내렸다고 볼 수 없다. 그래도 분명 강한 연관성은 있다.

```python
df1.columns

df1.show()
```
```
['Movie_Title', 'Genre', 'Num_Views']

+--------------------+------------------+---------+
|         Movie_Title|             Genre|Num_Views|
+--------------------+------------------+---------+
|        Finding Nemo|Animated Childrens|       12|
|           Toy Story|Animated Childrens|        6|
|            Iron Man|            Action|        1|
|     Captain America|            Action|        1|
|     The Incredibles|Animated Childrens|        9|
|              Frozen|Animated Childrens|       22|
|The Shawshank Red...|             Drama|        2|
|  Rabbit Proof Fence|             Drama|        2|
|Searching for Sug...|       Documentary|        3|
|              Powder|             Drama|        1|
|        The Fugitive|            Action|        2|
+--------------------+------------------+---------+
```

#### Ratings data types

```python
# Group the data by "Genre"
markus_ratings.groupBy("Genre").sum().show()
```
```
+------------------+--------------+
|             Genre|sum(Num_Views)|
+------------------+--------------+
|             Drama|             5|
|       Documentary|             3|
|            Action|             4|
|Animated Childrens|            49|
+------------------+--------------+
```

#### Confirm understanding of latent features

Matrix `P` is provided here. Its columns represent movies and its rows represent several latent features. Matrix `Pi` contains a rough approximation of what these latent features could represent. (Row latent feature name은 인위적으로 명시한 것이다. 사실, 아무도 정확히 명명하기는 어렵다.)

```python
# Examine matrix P using the .show() method
P.show()

# Examine matrix Pi using the .show() method
Pi.show()
```
```
+--------+------------+--------+---------+------------+------+----------+
|Iron Man|Finding Nemo|Avengers|Toy Story|Forrest Gump|Wall-E|Green Mile|
+--------+------------+--------+---------+------------+------+----------+
|     0.2|         2.4|     0.1|      2.4|           0|   2.5|         0|
|     1.5|         1.4|     1.4|      1.3|         1.8|   1.8|       2.5|
|     2.5|         1.1|     2.4|      0.9|         0.2|   0.9|      0.09|
|     1.9|           2|     1.5|      2.2|         1.2|   0.3|      0.01|
|       0|           0|       0|      2.3|         2.2|     0|       2.5|
+--------+------------+--------+---------+------------+------+----------+

+---------+--------+------------+--------+---------+------------+------+----------+
| Lat Feat|Iron Man|Finding Nemo|Avengers|Toy Story|Forrest Gump|Wall-E|Green Mile|
+---------+--------+------------+--------+---------+------------+------+----------+
| Animated|     0.2|         2.4|     0.1|      2.4|           0|   2.5|         0|
|    Drama|     1.5|         1.4|     1.4|      1.3|         1.8|   1.8|       2.5|
|Superhero|     2.5|         1.1|     2.4|      0.9|         0.2|   0.9|      0.09|
|   Comedy|     1.9|           2|     1.5|      2.2|         1.2|   0.3|      0.01|
|Tom Hanks|       0|           0|       0|      1.8|         2.2|     0|       2.5|
+---------+--------+------------+--------+---------+------------+------+----------+
```

### 2. How does ALS work?

ALS가 row-lank matrix factorization을 하고 난 뒤에, 이를 기반으로 collaborative filtering을 하는 것이다. 처음엔 user latent features를 학습하고, 그 다음엔 movie latent features를 학습한다. 그리고 이를 반복한다. 매 iteration마다 RMSE를 측정할 때는 missing value는 신경쓰지 않는다. 

하나의 거대한 matrix를 두 개의 factor matrices 로 변환하고, ALS로 iterative하게 학습하면, missing value들을 모두 prediction할 수 있다. 하나의 rating이 모든 user, 그리고 모든 movie에 영향을 주기 때문이다. 그리고 두 개의 factor matrices를 통해 user preferences, movie preferences를 분석할 수도 있다. 벡터 유사도 비교.

#### Matrix Multiplication

```python
# Use the .head() method to view the contents of matrices a and b
print("Matrix A:")
print (a.head())

print("Matrix B:")
print (b.head())

# Complete the matrix with the product of matrices a and b
product = np.array([[10,12], [15,18]])

# Run this validation to see how your estimate performs
product == np.dot(a,b)
```
```
Matrix A:
     0  1
One  2  2
Two  3  3
Matrix B:
     0  1
One  1  2
Two  4  4

Out[1]: 
array([[ True,  True],
       [ True,  True]])
```

#### Matrix Factorization

Matrix `G` is provided here as a Pandas dataframe. Look at the possible factor matrices `H`, `I`, and `J` (also Pandas dataframes), and determine which two matrices will produce the matrix G when multiplied together.

```python
# Take a look at Matrix G using the following print function
print("Matrix G:")
print(G)

# Take a look at the matrices H, I, and J and determine which pair of those matrices will produce G when multiplied together. 
print("Matrix H:")
print(H)
print("Matrix I:")
print(I)
print("Matrix J:")
print(J)

# Multiply the two matrices that are factors of the matrix G
prod = np.matmul(H,J)
print(G == prod)
```
```
Matrix G:
   0  1
0  6  6
1  3  3
Matrix H:
   0  1
0  2  2
1  1  1
Matrix I:
   0  1
0  3  3
1  3  3
Matrix J:
   0  1
0  1  1
1  2  2

      0     1
0  True  True
1  True  True
```

#### Non-Negative Matrix Factorization

It's possible for one matrix to have two equally close factorizations where one has all positive values and the other has some negative values.

The matrix `M` has been factored twice using two different factorizations. Take a look at each pair of factor matrices `L` and `U`, and `W` and `H` to see the differences. Then use their products to see that they produce essentially the same product.

```python
# View the L, U, W, and H matrices.
print("Matrices L and U:") 
print(L)
print(U)

print("Matrices W and H:")
print(W)
print(H)

# Calculate RMSE for LU
print("RMSE of LU: ", getRMSE(LU, M))

# Calculate RMSE for WH
print("RMSE of WH: ", getRMSE(WH, M))
```
```
Matrices L and U:
      0         1         2  3
0  1.00  0.000000  0.000000  0
1  0.01 -0.421053  0.098316  1
2  1.00  0.000000  1.000000  0
3  0.10  1.000000  0.000000  0
   0     1      2         3
0  1  2.00  1.000  2.000000
1  0 -0.19 -0.099 -0.198000
2  0  0.00  1.000 -1.000000
3  0  0.00  0.000  0.194947

Matrices W and H:
      0     1     2     3
0  2.61  0.24  0.00  0.12
1  0.00  0.05  0.02  0.17
2  1.97  0.00  0.58  0.83
3  0.05  0.00  0.00  0.00
      0     1     2     3
0  0.38  0.65  0.34  0.41
1  0.00  1.20  0.15  3.72
2  0.42  1.09  1.38  0.07
3  0.00  0.11  0.65  0.17

RMSE of LU:  0.072
RMSE of WH:  0.072
```

#### Estimating Recommendations

Use your knowledge of matrix multiplication to determine which movie will have the highest recommendation for `User_3`. The ratings matrix has been factorized into `U` and `P` with ALS.

```
# View left factor matrix
print(U)
        U_LF_1  U_LF_2  U_LF_3  U_LF_4
User_1    0.80    0.01    0.30     0.8
User_2    0.40    0.01    0.06     0.2
User_3    0.05    2.10    0.01     2.2
User_4    0.30    0.01    0.20     0.2
User_5    0.10    1.50    0.90     0.0
User_6    0.00    0.03    0.40     0.5
User_7    0.01    0.02    0.66     0.4
User_8    0.90    0.70    0.00     1.0
User_9    1.00    2.00    0.04     0.2

In [2]: # View right factor matrix
        print(P)
        Movie_1  Movie_2  Movie_3  Movie_4
P_LF_1      0.5      0.1      0.4     1.10
P_LF_2      0.2      2.0      0.0     0.01
P_LF_3      0.3      1.9      0.6     0.90
P_LF_4      1.0      0.2      1.0     0.89
```

```python
# Multiply factor matrices
UP = np.matmul(U,P)

# Convert to pandas DataFrame
print(pd.DataFrame(UP, columns = P.columns, index = U.index))
```
```
        Movie_1  Movie_2  Movie_3  Movie_4
User_1      NaN      NaN      NaN      NaN
User_2      NaN      NaN      NaN      NaN
User_3      NaN      NaN      NaN      NaN
User_4      NaN      NaN      NaN      NaN
User_5      NaN      NaN      NaN      NaN
User_6      NaN      NaN      NaN      NaN
User_7      NaN      NaN      NaN      NaN
User_8      NaN      NaN      NaN      NaN
User_9      NaN      NaN      NaN      NaN
```

#### RMSE As ALS Alternates

As you know, ALS will alternate between the two factor matrices, adjusting their values each time to iteratively come closer and closer to approximating the original ratings matrix. 

Matrix `T` is a ratings matrix, and matrices `F1`, `F2`, `F3`, `F4`, `F5`, and `F6` are the respective products of ALS after iterating 2, 3, 4, 5, and 6 times respectively. Follow the instructions below to see how the RMSE changes as ALS iterates.

```python
# Use getRMSE(preds, actuals) to calculate the RMSE of matrices T and F1.
getRMSE(F1, T)

# Create list of F2, F3, F4, F5, and F6
Fs = [F2, F3, F4, F5, F6]

# Calculate RMSEs for F2 - F6
getRMSEs(Fs, T)
```
```
F1: 2.4791263858912522
F2: 0.4389326310548279
F3: 0.17555006757053257
F4: 0.15154042416388636
F5: 0.13191130368008455
F6: 0.04533823201006271
```

#### Correct format and distinct users

Take a look at the `R` dataframe. Notice that it is in conventional or "wide" format with a different movie in each column. Also notice that the `User`'s and movie names are not in integer format. 

* Create a dataframe called `users` that contains all the `.distinct()` users from the dataframe and, repartition the dataframe into one partition using the `.coalesce(1)` method.
* Use the `monotonically_increasing_id()` method inside of `withColumn()` to create a new column in the users dataframe that contains a unique integer for each user. Call this column `userId`. Be sure to call the `.persist()` method on the final dataframe to ensure the new integer IDs persist.

```python
# Import monotonically_increasing_id and show R
from pyspark.sql.functions import monotonically_increasing_id
R.show()

# Use the to_long() function to convert the dataframe to the "long" format.
ratings = to_long(R)
ratings.show()

# Get unique users and repartition to 1 partition
users = ratings.select("User").distinct().coalesce(1)

# Create a new column of unique integers called "userId" in the users dataframe.
users = users.withColumn("userId", monotonically_increasing_id()).persist()
users.show()
```
```
+----------------+-----+----+----------+--------+
|            User|Shrek|Coco|Swing Kids|Sneakers|
+----------------+-----+----+----------+--------+
|    James Alking|    3|   4|         4|       3|
|Elvira Marroquin|    4|   5|      null|       2|
|      Jack Bauer| null|   2|         2|       5|
|     Julia James|    5|null|         2|       2|
+----------------+-----+----+----------+--------+

+----------------+----------+------+
|            User|     Movie|Rating|
+----------------+----------+------+
|    James Alking|     Shrek|     3|
|    James Alking|      Coco|     4|
|    James Alking|Swing Kids|     4|
|    James Alking|  Sneakers|     3|
|Elvira Marroquin|     Shrek|     4|
|Elvira Marroquin|      Coco|     5|
|Elvira Marroquin|  Sneakers|     2|
|      Jack Bauer|      Coco|     2|
|      Jack Bauer|Swing Kids|     2|
|      Jack Bauer|  Sneakers|     5|
|     Julia James|     Shrek|     5|
|     Julia James|Swing Kids|     2|
|     Julia James|  Sneakers|     2|
+----------------+----------+------+

+----------------+------+
|            User|userId|
+----------------+------+
|Elvira Marroquin|     0|
|      Jack Bauer|     1|
|    James Alking|     2|
|     Julia James|     3|
+----------------+------+
```

#### Assigning integer id's to movies

* Use the `.select()` and the `.distinct()` methods to extract all unique `Movies` from the `ratings` dataframe.
* Repartition the `movies` dataframe to one partition using `coalesce()`.
* Complete the partial code provided to assign unique integer IDs to each movie. Name the new column `movieId` and call the `.persist()` method on the resulting dataframe.
* Join the `ratings` dataframe to the `users` dataframe and subsequently to the `movies` dataframe. Call the result `movie_ratings`.


```python
# Extract the distinct movie id's
movies = ratings.select("Movie").distinct() 

# Repartition the data to have only one partition.
movies = movies.coalesce(1) 

# Create a new column of movieId integers. 
movies = movies.withColumn("movieId", monotonically_increasing_id()).persist() 

# Join the ratings, users and movies dataframes
movie_ratings = ratings.join(users, "User", "left").join(movies, "Movie", "left")
movie_ratings.show()
```
```
+----------+----------------+------+------+-------+
|     Movie|            User|Rating|userId|movieId|
+----------+----------------+------+------+-------+
|     Shrek|    James Alking|     3|     2|      3|
|      Coco|    James Alking|     4|     2|      1|
|Swing Kids|    James Alking|     4|     2|      2|
|  Sneakers|    James Alking|     3|     2|      0|
|     Shrek|Elvira Marroquin|     4|     0|      3|
|      Coco|Elvira Marroquin|     5|     0|      1|
|  Sneakers|Elvira Marroquin|     2|     0|      0|
|      Coco|      Jack Bauer|     2|     1|      1|
|Swing Kids|      Jack Bauer|     2|     1|      2|
|  Sneakers|      Jack Bauer|     5|     1|      0|
|     Shrek|     Julia James|     5|     3|      3|
|Swing Kids|     Julia James|     2|     3|      2|
|  Sneakers|     Julia James|     2|     3|      0|
+----------+----------------+------+------+-------+
```



















