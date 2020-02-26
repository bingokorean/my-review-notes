# DataCamp

### Contents

* [Introduction to PySpark](#Introduction-to-PySpark)
* [Big Data Fundamentals with PySpark](#Big-Data-Fundamentals-with-PySpark)





<br>




## Introduction to PySpark

### 1. Getting to know PySpark

Pyspark에서 Spark cluster와 연결하는 법은 해당 클래스 객체를 생성하는 일은 'SparkContext 클래스' 에서 담당한다. <br>
클러스터의 속성은  SparkContext 객체 생성시 인자를 통해 명시한다.

클러스터 속성들은 'SparkConf 클래스' 를 통해 생성될 수 있다.

간단한 계산은 오히려 spark에서 오래 걸릴 수도 있다. <br>
spark는 대용량 데이터에 맞는 복잡한 계산으로 최적화되어 있기 때문이다.

```
# SparkContext 를 sc 에 생성했다고 가정.

# Verify SparkContext
print(sc)

# Print Spark version
print(sc.version)
>>>
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

```
# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession

# Create my_spark
my_spark = SparkSession.builder.getOrCreate()

# Print my_spark
print(my_spark)
>>>
<pyspark.sql.session.SparkSession object at 0x7f0c0ffaa128>
```

SparkSession 을 만들면, 클러스터에 있는 데이터를 확인할 수 있다. <br>
SparkSession은 catalog 속성을 가진다. 이는 cluster 안에 있는 모든 데이터의 리스트를 가진다. <br>
catalog.listTables() 함수로 cluster에 있는 모든 테이블의 이름을 확인할 수 있다.

```
# Print the tables in the catalog
print(spark.catalog.listTables())
>>>
[Table(name='flights', database=None, description=None, tableType='TEMPORARY', isTemporary=True)]
```

DataFrame의 장점 중 하나는 SQL 쿼리를 Spark cluster에 날릴 수 있다는 점이다.

```
query = "FROM flights SELECT * LIMIT 10"

# Get the first 10 rows of flights
flights10 = spark.sql(query)

# Show the results
flights10.show()
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

```
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"

# Run the query
flight_counts = spark.sql(query)

# Convert the results to a pandas DataFrame
pd_counts = flight_counts.toPandas()

# Print the head of pd_counts
print(pd_counts.head())
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

```
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

```
# Don't change this file path
file_path = "/usr/local/share/datasets/airports.csv"

# Read in the airports data
airports = spark.read.csv(file_path, header=True)

# Show the data
airports.show()
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

```
# Create the DataFrame flights
flights = spark.table("flights")

# Add duration_hrs
flights = flights.withColumn("duration_hrs", flights.air_time/60)

# Show the head
flights.show()
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

```
# Filter flights by passing a string
long_flights1 = flights.filter("distance > 1000") (1) WHERE clause of a SQL query 

# Filter flights by passing a column of boolean values
long_flights2 = flights.filter(flights.distance > 1000) (2) a column of boolean values

# Print the data to check they're equal
long_flights1.show()
long_flights2.show()
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

```
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

```
# Define avg_speed
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")

# Select the correct columns
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)

# Create the same table using a SQL expression
speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed")
```

#### Aggregating

보편적인 aggregation method인 `.min()`, `.max()`, 그리고 `.count()` 는 `GroupedData` method라 불린다. 이들은 `.groupBy()` DataFrame method를 부름과 동시에 생성된다. Aggregate를 하기 전에 groupby를 해줘야 한다. 여기 groupBy()에서는 인자가 없으므로 전체 범위라고 보면 된다.

```
# Find the shortest flight from PDX in terms of distance
flights.filter(flights.origin == "PDX").groupBy().min("distance").show()

# Find the longest flight from SEA in terms of air time
flights.filter(flights.origin == "SEA").groupBy().max("air_time").show()
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

```
# Average duration of Delta flights
flights.filter(flights.carrier == "DL").filter(flights.origin == "SEA").groupBy().avg("air_time").show()

# Total hours in the air
flights.withColumn("duration_hrs", flights.air_time/60).groupBy().sum("duration_hrs").show()
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

```
# Group by tailnum
by_plane = flights.groupBy("tailnum")

# Number of flights each plane made
by_plane.count().show()

# Group by origin
by_origin = flights.groupBy("origin")

# Average duration of flights from PDX and SEA
by_origin.avg("air_time").show()

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
 
```
# Import pyspark.sql.functions as F
import pyspark.sql.functions as F

# Group by month and dest
by_month_dest = flights.groupBy("month", "dest")

# Average departure delay by month and destination
by_month_dest.avg("dep_delay").show()

# Standard deviation of departure delay
by_month_dest.agg(F.stddev("dep_delay")).show()

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

```
# Examine the data
airports.show()

# Rename the faa column
airports = airports.withColumnRenamed("faa", "dest")

# Join the DataFrames
flights_with_airports = flights.join(airports, on="dest", how="leftouter")

# Examine the new DataFrame
flights_with_airports.show()

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

```
# Create an RDD from a list of words
RDD = sc.parallelize(["Spark", "is", "a", "framework", "for", "Big Data processing"])

# Print out the type of the created object
print("The type of RDD is", type(RDD))
>>>
The type of RDD is <class 'pyspark.rdd.RDD'>
```

#### RDDs from External Datasets

```
# Print the file_path
print("The file_path is", file_path)

# Create a fileRDD from file_path
fileRDD = sc.textFile(file_path)

# Check the type of fileRDD
print("The file type of fileRDD is", type(fileRDD))

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

```
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

```
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

```
# Print the first 10 observations 
people_df.show(10)

# Count the number of rows 
print("There are {} rows in the people_df DataFrame.".format(people_df.count()))

# Count the number of columns and their names 
print("There are {} columns in the people_df DataFrame and their names are {}".format(len(people_df.columns), people_df.columns))

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

```
# Select name, sex and date of birth columns
people_df_sub = people_df.select('name', 'sex', 'date of birth')

# Print the first 10 observations from people_df_sub
people_df_sub.show(10)

# Remove duplicate entries from people_df_sub
people_df_sub_nodup = people_df_sub.dropDuplicates()

# Count the number of rows
print("There were {} rows before removing duplicates, and {} rows after removing duplicates".format(people_df_sub.count(), people_df_sub_nodup.count()))

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

```
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

```
# Create a temporary table "people"
people_df.createOrReplaceTempView("people")

# Construct a query to select the names of the people from the temporary table "people"
query = '''SELECT name FROM people'''

# Assign the result of Spark's query to people_df_names
people_df_names = spark.sql(query)

# Print the top 10 names of the people
people_df_names.show(10)

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

```
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

```
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

```
# Load the Dataframe 
fifa_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Check the schema of columns 
fifa_df.printSchema()

# Show the first 10 observations
fifa_df.show(10)

# Print the total number of rows
print("There are {} rows in the fifa_df DataFrame".format(fifa_df.count()))

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

```
# Create a temporary view of fifa_df
fifa_df.createOrReplaceTempView('fifa_df_table')

# Construct the "query"
query = '''SELECT Age FROM fifa_df_table WHERE Nationality == "Germany"'''

# Apply the SQL "query"
fifa_df_germany_age = spark.sql(query)

# Generate basic statistics
fifa_df_germany_age.describe().show()

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

```
fifa_df_germany_age_pandas = fifa_df_germany_age.toPandas()

# Plot the 'Age' density of Germany Players
fifa_df_germany_age_pandas.plot(kind='density')
plt.show()

>>>

...
```





