# Data Analytics with Spark using Python

빅데이터 분석을 위한 파이썬, 스파크 활용법 <br>
제프리 에이븐 (송주경 옮김)


### Contents

<div id='contents'/>

1. [빅데이터, 하둡 및 스파크 소개](#1.)
2. [스파크 배포](#2.)
3. [스파크 클러스터 아키텍처의 이해](#3.)
4. [스파크 프로그래밍 기초 학습](#4.)
5. [스파크 코어 API를 사용한 고급 프로그래밍](#5.)
6. 스파크로 SQL 및 NoSQL 프로그래밍하기
7. 스파크를 사용한 스트림 처리 및 메시징
8. 스파크를 사용한 데이터 과학 및 머신 러닝 소개


#### Practices
* [맵리듀스 및 Word Count 연습](#맵리듀스-및-Word-Count-연습)
* [Broadcast 변수 및 accumulator 사용 연습](#Broadcast-변수-및-accumulator-사용-연습)

<br>


#### 옮긴이의 말

* 하둡의 맵리듀스(MapReduce)는 슈퍼컴퓨터 없이 여러 대의 서버를 연결해 빅데이터 분석을 가능하게 한 오픈소스 프레임워크
* 스파크는 맵리듀스처럼 분산 처리를 수행하지만, 메모리를 활용하여 빠르게 데이터를 처리하는 것이 특징
* 이에 스파크는 스트리밍 데이터 처리같은 실시간 처리와 머신러닝을 통한 애플리케이션과의 복합적 운영이 필요할 때 적합하다! (ex. 실시간 타겟마켓팅, 고객 분석 및 추천)

<br>

<div id='1.'/>

## 1. 빅데이터, 하둡 및 스파크 소개

### 1.1. 빅데이터, 분산 컴퓨팅 및 하둡 소개

* 하둡 프로젝트의 핵심 개념
   * 데이터 지역성(data locality)
   * 비공유(shared nothing)
   * 맵리듀스(MapReduce)
* 하둡의 간단한 역사 (검색 엔진 기업(구글)에서 발달)
   * The Google File System (2003)
   * MapReduce: Simplified Data Processing on Large Clusters (2004)
* 데이터 생선량의 급격한 증가로 하둡이 곽강을 받기 시작. Spark, Kafka(메시징 시스템), HBase, Cassandra 등 NOSQL에 대한 논의가 이뤄졌는데, 이 모든 것은 하둡에서 시작되었음.


#### 하둡

* 데이터 지역성(data locality)이라는 개념을 바탕을 둔 데이터 저장 및 처리 플랫폼.
* 데이터 지역성(data locality)이란? 요청한 데이터를 원격 처리 시스템이나 호스트로 보내고 처리하는 기존의 방식 대신 데이터가 있는 곳으로 이동해서 계산하는 방식.
* 빅데이터의 경우 컴퓨팅 시간에 많은 양의 데이터가 네트워크를 통해 이동하는 시간이 매우 크거나 경우에 따라서 불가능할 수도 있음.
* 하둡은 대용량 데이터가 비공유(shared nothing) 접근을 사용하는 클러스터의 노드에서 지역적으로 처리할 수 있음.
* 각 노드는 다른 노드들과 통신할 필요 없이 전체 데이터의 훨씬 작은 부분을 독립적으로 처리할 수 있음.
* 이는 분산 파일 시스템의 구현을 통해 가능함.
* 'Schema-on-read': 하둡은 기록 연산과 관련된 스키마가 없음. 이는 비구조화 문서, 반구조화 JSON, XML, DBMS의 잘 구조화된 문서 범위에 이르는 광범위한 데이터를 저장하고 처리할 수 있다는 뜻임. ('Schema-on-read'는 'Schema-on-write'와 대조적)
* HBase, 카산드라와 같은 NoSQL 플랫폼도 'Schema-on-read'.
* 스키마는 INSERT, UPDATE 또는 UPSERT 작업 시 미리 개념 정리가 되어 시스템에 적용됨.
* 하둡에서 쓰기 작업을 수행하는 동안에 스키마는 해석되지 않으므로 인덱스, 통계, 기타 구조가 없음. 단, 데이터 지역성(data locality)이 필요. 
* 하둡은 큰 문제를 작은 문제의 집합으로 나누고 연산하며, 데이터 지역성(data locality)과 비공유(shared nothing) 개념을 적용함.

#### 하둡의 핵심 구성 요소

* 하둡의 구성 요소
   * 하둡 분산 파일 시스템(HDFS; Hadoop Distributed File System): HDFS는 하둡의 스토리지 서브시스템 
   * YARN(Yet Another Resource Negotiator): YARN은 하둡의 리소스 스케줄링 서브시스템
* 각 구성 요소는 자체 클러스터에서 서로 독립적으로 작동할 수 있음
* HDFS 클러스터와 YARN 클러스터가 서로 결합된 두 시스템의 조합을 하둡 클러스터라고 함
* 스파크는 하둡의 이러한 핵심 구성 요소를 모두 활용할 수 있음
* 클러스터 용어
   * 클러스터는 연산이나 프로세싱 함수를 수행하기 위해 함께 작동하는 시스템 모음. 클러스터 내의 개별 서버는 노드(node)임
   * 클러스터에는 여러 토폴로지 및 통신 모델이 있는데, 그중 하나가 master/slave 모델임
   * master/slave 모델은 한 프로세스가 하나 이상의 다른 프로세스를 제어하는 통신 모델임 (HDFS, YARN 클러스터)
* Flume이나 Sqoop과 같은 데이터 처리 프로젝트 또는 Pig나 Hive와 같은 데이터 분석 툴처럼 하둡과 상호 작용하거나 통합하는 프로젝트를 하둡 '에코시스템' 프로젝트라고 한다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_1_1.png" width="60%" height="60%"></p>


#### HDFS: 파일, 블록 및 메타데이터

* HDFS는 클러스터의 하나 이상의 노드(node)에 파일이 분산돼 있는 블록(block)으로 구성된 가상(virtual) 파일시스템이다.
* Ingestion 프로세스
   * 파일시스템에 데이터를 업로드할 때 구성된 블록의 크기에 따라 무작위로 파일을 나눈다.
   * 그 후, 클러스터에 있는 노드들에 걸쳐 블록을 분산 및 복제해서 내결함성(fault tolerance)을 달성하고, 데이터에 계산을 가져오는 목적으로 설계된 로컬에서 데이터를 처리할 수 있도록 한다.
* HDFS 블록은 DataNode 라는 불리우는 'slave 노드 HDFS 클러스터 프로세스'에 저장 및 관리된다.
* DataNode 프로세스는 HDFS 클러스터의 하나 이상의 노드에서 실행되는 'HDFS slave 노드 데몬'이다.
* DataNode는 블록 스토리지 관리, 데이터 읽기 및 쓰기를 위한 액세스 및 데이터 ingestion 프로세스의 일부인 블록 복제를 관리한다 (다음 그림 참조)

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_1_2.png" width="60%" height="60%"></p>

* 일반적으로 많은 호스트는 완전히 분산된 하둡 클러스터에서 데이터 노드 프로세스를 실행한다.
* 데이터 노드 프로세스는 하둡에 구축된 스파크 응용 프로그램을 위한 분산 스파크 작업자(worker) 프로세스에 파티션 형식의 입력 데이터를 제공한다.
<br>

* 파일시스템과 가상 디렉토리, 파일 및 파일을 구성하는 물리적 블록에 대한 정보는 파일 시스템 메타데이터(metadata)에 저장된다.
* 파일시스템 메타데이터는 네임노드(NameNode)라는 HDFS 마스터 노드 프로세스의 상주 메모리에 저장된다.
* HDFS 클러스터의 네임노드는 관계형 데이터베이스 트랜잭션 로그와 유사한 저널링(journaling) 함수를 통해 메타데이터에 대한 내구성을 제공하고, HDFS 클라이언트에 읽기 및 쓰기 작업을 위한 블록 위치를 제공한다.
* 클라이언트는 데이터 연산을 위해 데이터노드(DataNode)와 직접 통신한다.
* 그림 1.3은 HDFS 읽기 작업 구조를, 그림 1.4는 HDFS의 쓰기 작업의 구조를 나타낸다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_1_3.png" width="60%" height="60%"></p>

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_1_4.png" width="60%" height="60%"></p>

#### YARN 을 이용한 응용 스케줄링

* 하둡은 일반적으로 HDFS에서 데이터를 가져오고, HDFS에 데이터를 기록한다. YARN은 이런 하둡의 데이터 처리를 제어하고 조율한다.
* YARN 클러스터 아키텍쳐는 HDFS의 master/slave 클러스터 프레임워크와 같다.
   * 여기서 HDFS는 리소스 매니저(Resource Manager)라는 마스터 노드 데몬과
   * 클러스터의 작업자(Worker)나 slave 노드에서 실행되는 노드 매니저(Node Manager)라는 하나 이상의 slave 노드 데몬을 포함한다.
<br>

* 리소스 매니저는 클러스터에서 실행 중인 응용 프로그램에 클러스터 컴퓨팅 리소스를 부여한다.
   * 리소스는 컨테이너(container)라는 단위로 미리 정의된 CPU 코어와 메모리 조합이다.
   * 컨테이너는 최소 임곗값 및 최대 임곗값을 포함해 할당되고 클러스터에서 구성되며, 하나 이상의 프로세스 전용 리소스를 분리하는 데 사용된다.




...









#### YARN을 이용한 응용 스케줄링

* 하둡은 일반적으로 HDFS에서 데이터를 가져오고, HDFS에 데이터를 기록한다.
* YARN은 이런 하둡의 데이터 처리를 제어하고 조율한다.
* YARN 클러스터 아키텍쳐는 HDFS의 master/slave 클러스터 프레임워크와 같다.
* 여기서, HDFS는 ResourceManager라는 master 노드 데몬과 클러스터의 Worker나 slave 노드에서 실행되는 NodeManager라는 하나 이상의 slave 노드 데몬을 포함한다.
* ResourceManager는 클러스터에서 실행 중인 응용 프로그램에 클러스터 컴퓨팅 리소스를 부여한다.





...



### 1.2. 아파치 스파크 소개

* 맵리듀스 구현의 주요 단점은 맵(Map)과 리듀스(Reduce) 처리 단계 사이의 중간 데이터가 디스크에 잔류한다는 것이다.
* 맵리듀스 대안으로 스파크는 탄력적인 분산 데이터 집합(**RDD**; Resilient Distributed Dataset)이라는 분산형, 내결함성, 인메모리 구조를 구현한다.
* 스파크는 여러 컴퓨터에서 메모리 사용을 극대화해 전반적인 성능을 크게 향상시킨다.
* 스파크의 이러한 인메모리 구조의 재사용은 반복적인 머신러닝 작업 및 대화형 쿼리에 적합하다.
* 스파크는 JVM과 자바 런타임 위에 구축된 스칼라(Scala)로 작성되었다.
* 스파크는 개발자가 하드웨어 오류와 같은 인프라 또는 환경 문제가 아닌 논리에 집중할 수 있도록 높은 수준의 API 및 내결함성 프레임워크를 제공해 복잡한 다단계 데이터 처리 루틴을 만들 수 있게 한다.

#### 스파크 프로그램의 제출 유형

* 스파크 프로그램은 대화식 또는 배치 작업 (미니 배치 및 마이크로 배치 작업 포함)으로 실행할 수 있다.
* 대화형 프로그래밍 shell은 파이썬과 스칼라에서 사용할 수 있다.
* 비대화형 응용 프로그램은 spark-submit 명령을 사용해 제출할 수 있다.

#### 스파크 응용 프로그램의 입력/출력 유형

* 스파크는 주로 하둡에서 데이터를 처리하는 데 사용되며, 다음과 같은 다양한 소스 및 대상 시스템과도 함께 사용한다.
   * 로컬 또는 네크워크 파일 시스템
   * 아마존 S3 또는 Ceph와 같은 객체 저장소
   * 관계형 데이터베이스 시스템
   * 카산드라, HBase 등을 포함한 NoSQL 스토어
   * Kafka 같은 메시징 시스템

#### 스파크 RDD

* 스파크 RDD는 스파크 응용 프로그램의 기본 데이터 추상화 구조이다.
* 스파크와 다른 클러스터 컴퓨팅 프레임워크 사이의 주요 차별화 요소 중 하나이다.
* 스파크 RDD는 클러스터에 분산된 인메모리 데이터 모음으로 간주할 수 있다.
* 스파크 프로그램은 입력 데이터를 RDD에 로드하고 RDD를 후속 RDD로 변환한 다음, 최종 RDD에서 응용 프로그램의 최종 출력을 저장하거나 표시하는 것으로 구성되며, 스파크 코어 API를 사용한다.

#### 스파크와 하둡

* 하둡과 스파크는 비공유 및 데이터 지역성과 같은 핵심 병렬 처리 개념의 정립과 구현 과정에서 서로 밀접한 관련을 가진다.
* 하둡(일반적으로 HDFS)의 데이터 처리 프레임워크로 스파크를 배포할 수 있다.
* 스파크는 기본적으로 HDFS에서 파일을 읽고 HDFS로 데이터를 쓰는 기능이 있다.

#### 스파크의 리소스 스케줄러로서의 YARN

* YARN은 스파크 응용 프로그램에서 가장 일반적으로 사용되는 프로세스 스케줄러 중 하나다.
* YARN은 대개 하둡 클러스터의 HDFS와 함께 있으므로 스파크 응용 프로그램을 관리하기 편리한 플랫폼이다.
* YARN은 하둡 클러스터의 분산 노드에서 사용할 수 있는 컴퓨팅 리소스를 관리하므로
   * 스파크 처리 단계를 가능하면 병렬로 실행할 수 있도록 예약할 수 있다.
   * 또한, 스파크 응용 프로그램의 입력 소스로 HDFS를 사용하는 경우 YARN은 데이터 지역성을 최대한 활용하도록 맵 작업을 예약할 수 있다.
   * 이는 처리 초기 단계에서 네트워크를 통해 전송해야 하는 데이터의 양을 최소화할 수 있다.


### 1.3. 파이썬을 이용한 함수 프로그래밍

* 파이썬은 멀티 패러다임 프로그래밍 언어로서, 명령 지향 프로그래밍 방식 패러다임과 객체 지향 및 함수 패러다임을 완벽하게 지원하는 프로그래밍 패러다임을 결합한다.

#### 파이썬 함수 프로그래밍에서 사용되는 데이터 구조

* 스파크의 파이썬 RDD는 파이썬 객체의 분산 컬렉션을 단순하게 표현한 것으로서 파이썬에서 사용할 수 있는 다양한 데이터 구조를 먼저 이해해야 한다.

##### List 

* 리스트는 세 가지 주요 함수 프로그래밍(map(), reduce(), filter())뿐만 아니라 count(), sort() 등을 포함한 기타 기본 제공 메소드를 지원한다.
* 스파크 RDD는 본질적으로 파이썬 리스트를 표현한다.
* map() 함수는 입력 리스트에서 작동하고 새 리스트를 반환한다.

```python
tempc = [38.4, 19.2, 12.8, 9.6]
tempc = map(lambda x: (float(9)/5)*x + 32, tempc)
print(tempc)
```
```
>>>
[101.12, 66.56, 55.0400000000006, 49.28]
```

* 파이썬 리스트는 기본적으로 변경할 수 있지만, 스파크의 파이썬 RDD에 포함된 리스트 객체는 변경할 수 없다. (스파크 RDD에서 생성된 객체의 경우도 마찬가지이다)
* set는 파이썬에서 사용할 수 있는 객체 유형으로, 설정된 수학적 추상화를 기반으로 한다. set는 union(), intersection() 등과 같은 일반적인 수학적 집합 연산을 지원하며, 순서가 정해지지 않은 고유한 값을 모음이다.

##### Tuple

* 단순하게 튜플은 고정 리스트와 비슷하다고 생각할 수 있지만, 그들은 서로 다른 구조를 가지며 목적도 매우 다르다.
* 튜플은 관계형 데이터베이스 테이블의 레코드와 유사하며, 각 레코드는 구조를 갖고 있다. 구조 안에 순서대로 위치한 각 필드는 모두 의미를 가진다. 리스트 객체의 순서는 기본적으로 변경할 수 있으므로 구조와 직접적인 관련이 없다.
* 튜플은 스파크 프로그래밍에서 키/값 쌍을 나타내기 위해 사용되므로 스파크의 필수적인 객체이다.

##### Dictionary

* 딕셔너리는 파이썬에서 순서가 정해지지 않은 변경 가능한 키/값 쌍이다.
* 리스트나 튜플과 달리 요소가 시퀀스에 순차적으로 액세스되므로 딕트의 요소는 키에 의해 액세스된다.
* 딕셔너리는 미리 정의된 스키마나 순서에 의존하지 않고 요소가 직접 설명한다.
* 딕셔너리 함수는 파이썬 RDD에서 고정 객체로 사용될 수 있다.

#### 파이썬 객체 직렬화

* 직렬화(serialization)는 같은 시스템이나 다른 시스템에서 객체를 압축 해제(비직렬화)할 수 있는 구조로 변환하는 프로세스이다.
* 직렬화 또는 데이터 직렬화 및 비직렬화는 배포된 모든 처리 시스템에 필수적인 기능이며, 하둡 및 스파크 프로젝트 전체에서 매우 중요하다.

##### JSON

* JSON(JavaScript Object Notation)은 일반적인 직렬화 형식이다. 거의 모든 프로그래밍 언어에서 지원하는 다양한 플랫폼에서 사용되고, 웹 서비스에서 반환되는 일반적인 응답 구조이다.
* JSON 객체는 키/값 쌍(딕셔너리) and/or 배열(리스트)로 구성되는데, 이들은 서로 내포될 수 있다.
* JSON 객체는 PySpark의 RDD에서 사용할 수 있다.

##### Pickle

* 피클은 파이썬의 독점적인 직렬화 메소드로 JSON보다 빠르다. 
* JSON은 일반적으로 상호 교환이 용이한 직렬화 형식으로 뛰어난 호환성을 가진다. 반면, 피클에서는 그 기능이 떨어진다.
* 파이썬의 pickle 모듈은 하나 이상의 객체들을 byte stream으로 변환해 전송, 저장 및 원래 상태로 재구성한다.
* PickleSerializer는 PySpark에서 객체를 피클링된 형식으로 로드하고 언피클 처리하는 데 사용된다. 여기에는 하둡의 SequenceFiles와 같이 다른 시스템에서 미리 직렬화된 객체를 읽고 파이썬에서 사용할 수 있는 형식으로 변환하는 작업이 포함된다.
* PySpark는 피클된 입출력 파일은 다루는 pickleFile과 saveAsPickleFile.pickleFile 메소드를 포함한다. 이 두 메소드는 PySpark 프로세스 간에 파일을 저장하고 전송하기 위한 효율적인 포맷이다.
* 피클은 개발자가 명시적으로 사용하는 것 외에도 파이썬에서 스파크 응용 프로그램을 실행할 때 많은 내부 스파크 프로세스에 의해 사용된다.

#### 파이썬 함수형 프로그래밍 기초

* 파이썬의 함수형 지원은 다음을 포함해 가능한 모든 함수 프로그래밍의 패러다임 특성을 구현한다.
   * first-class 객체로서의 함수와 프로그래밍의 기본 단위
   * 입출력 전용 함수
   * 고차원 함수 지원
   * 익명 함수 지원

##### 익명 함수 및 lambda 구문

* 익명 함수는 이름 없는 함수는 Lisp, Scala, JavaScript, Erlang, Clojure, Go 등의 함수형 프로그래밍 언어의 일관된 특징이다.
* 파이썬의 익명 함수는 def 키워드 대신에 lambda 구문을 사용해서 구현한다.
* 익명 함수는 입력 인수의 수에는 제한이 없지만, 하나의 값만 반환된다. 이 값은 다른 함수, 스칼라 값 또는 리스트 같은 데이터 구조일 것이다.
* 익명 함수의 진정한 힘은 스파크에서 작업하듯이 map(), reduce() 및 filter()와 같은 고차원 함수와 프로세싱 파이프라인의 단일 사용 함수를 함께 연결하기 시작하는 경우에 분명히 나온다.

```python
## 명명된 함수
def plusone(x): return x+1
plusone(1) 
# 2
type(plusone)
# <type 'function'>
plusone.func_name
# 'plusone'

## 익명 함수
plusonefn = lambda x: x+1
plusonefn(1)
# 2
type(plusonefn)
# <type 'function'>
plusonefn.func_name
# '<lambda>'
```

##### 고차원 함수

* 고차원 함수는 함수를 인수로 받아들이고 결과로 함수를 반환한다.
* 고차원 함수에는 map(), reduce(), filter()가 있는데, 이 함수들은 인수로 함수를 받아들인다.
* 함수를 리턴값으로 반환하는 함수는 고차원 함수로 간주되는데 이 특성은 비동기 프로그래밍에서 구현되는 콜백을 정의한다.

```python
# 스파크에서 고차원 함수의 예
lines = sc.textFile("file:///opt/spark/licences")
counts = lines.flatMap(lambda x: x.split(' ')) \
	.filter(lambda x: len(x) > 0) \
	.map(lambda x: (x, 1)) \
	.reduceByKey(lambda x, y: x+y) \
	.collect()

for (word, count) in counts:
	print(word, count)
```

##### 클로저(closure)

* 클로저는 인스턴스화된 시간에, 범위를 묶는 함수 객체다.
* 클로저 객체에는 함수가 작성될 때 사용된 외부 변수 또는 함수가 포함될 수 있다.
* 클로저는 범위를 묶어서 값을 기억한다.

```python
def generate_message(concept):
	def ret_message():
		return 'This is an example of ' + concept
	return ret_message

>>> call_func = generate_message('closure in Python')
>>> call_func
<function ret_message at 0x7fd138aa55f0>
>>> call_func()
'This is an example of closures in Python'

# 클로저 검사
>>> call_func.__closure__
(<cell at 0xfd138aaa638: str object at 0xfd138aaa638>,)
>>> type(call_func.__closure__[0])
<type 'cell'>
>>> call_func.__closure__[0].cell_contents
'closure in Python'

# 함수 삭제
del generate_message

# 클로저 재호출
call_func()
'This is an example of closures in Python' # 클로저 여전히 작동함!
```

* 위 코드에서 ret_message() 함수는 클로저이고, concept 값은 함수 범위로 묶인다.
* __closure__ 함수 멤버를 사용해 클로저에 관한 정보를 볼 수 있고, 함수에 포함된 참조는 셀의 튜플에 저장된다.
* 위 코드에 표시된 대로 cell_contents 함수를 사용해 셀 콘텐츠에 액세스할 수 있다.
* 클로저 개념을 증명하기 위해 외부 함수 generate_message를 삭제해 참조 함수인 call_func이 여전히 작동하는지 확인할 수 있다.
* 분산된 스파크 응용 프로그램에서 클로저가 중요한 이점을 가질 수 있으므로 클로저 개념을 파악하는 것은 매우 중요하다.
* 반면, 클로저는 함수가 어떻게 구성되고 호출되는지에 따라 부정적인 영향을 미칠 수도 있다.

[[top](#contents)]


<br>

<div id='2.'/>

## 2. 스파크 배포

### 2.1. 스파크 배포 모드

* 스파크에 대한 일반적인 배포 모드
   * 로컬 모드
   * 독립실행형(Standalone) 스파크
   * YARN(하둡)에서의 스파크
   * 메소스에서의 스파크
* 각 배포 모드는 스파크 런타임 아키텍쳐를 구현하는 것과 비슷하다. 단, 컴퓨팅 클러스터의 하나 또는 그 이상의 노드에서 리소스를 관리하는 방식만 다르다.
* YARN 또는 메소스와 같은 외부 스케줄러를 사용해 스파크를 배포하는 경우에는 로컬 모드로 실행하거나 스파크 독립실행형(Standalone) 스케줄러를 사용하면 된다. 단, 이 때 스파크 외부 종속성이 제거된다.
* 모든 스파크 배포 모드는 스트리밍 응용 프로그램을 포함하여 대화형(shell) 및 비대화형(배치) 응용 프로그램에서 사용할 수 있다.

#### 로컬 모드

* 모든 스파크 프로세스가 단일 시스템에서 실행되도록 한다. 
* 로컬 시스템의 코어 수를 임의로 선택해 사용한다.
* 새로 설치된 스파크를 테스트하는 빠른 방법이 될 수 있다. 작은 dataset에 대해 스파크 루틴을 신속하게 테스트할 수 있다.

```console
# 로컬 모드에서 스파크 작업 제출

$SPARK_HOME/bin/spark-submit \
--class org.apache.spark.examples.SparkPi \
--master local \

&SPARK_HOME/examples/jars/spark-examples*.jar 10
```

* 로컬 모드에서 사용할 코어 수는 local 명령문 뒤 괄호 안에 숫자로 나타낸다. (ex. 2개의 코어를 사용하려면 local[2], 시스템의 모든 코어를 사용하려면 local[`*`]로 표기함.)
* 로컬 모드에서 스파크를 실행할 때 로컬 시스템에 적절한 구성과 라이브러리를 사용할 수 있다면 로컬 파일 시스템의 모든 데이터와 HDFS, S3 또는 기타 파일 시스템의 데이터에 엑세스할 수 있다.
* 로컬 모드를 사용하면 빠르게 시작하고 실행할 수 있지만, production use case의 확장 및 효과 측면에서는 제한적이다.

#### 스파크 독립실행형(standalone)

* 스파크 독립실행형은 내장형 또는 '독립실행형' 스케줄러를 나타낸다.
* 독립실행형(standalone)은 말 그대로 클러스터 토폴로지와 아무 관련이 없다. 완전히 분산된 다중 노드 클러스터에서 독립실행형 모드로 스파크를 배포할 수도 있다. 이 경우 독립실행형은 외부 스케줄러가 필요없다는 뜻이다.
* 다중 호스트 프로세스 또는 서비스는 스파크 독립실행형 클러스터에서 실행되며, 각 서비스는 클러스터에서 실행 중인 지정된 스파크 응용 프로그램의 계획, 조정 및 관리에 중요한 역할을 한다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_2_1.png" width="60%" height="60%"></p>

* 위 그림은 완전히 분산된 스파크 독립실행형 참조 클러스터 토폴로지를 보여준다.
* URL 스키마로 스파크를 지정해 지정된 호스트 및 포트와 함께 응용 프로그램을 스파크 독립실행형 클러스터에 제출할 수 있다. 여기서 지정된 호스트 및 포트에서는 스파크 마스터 프로세스가 실행 중이다. 다음 코드는 이 예제를 보여준다.

```console
# 스파크 독립실행형 클러스터에 스파크 작업 제출

$SPARK_HOME/bin/spark-submit \
--class org.apache.spark.examples.SparkPi \
--master spark://mysparkmaster:7077 \

&SPARK_HOME/examples/jars/spark-examples*.jar 10
```

* 스파크 독립실행형을 사용하면 종속성이나 환경에 대한 깊은 고민 없이 빠르게 시작하고 실행할 수 있다.
* 각 스파크 릴리즈에는 스파크 독립실행형 클러스터에서 호스트가 지정된 역할을 맡을 수 있게 하는 바이너리 및 구성 파일과 함께 시작하는 데 필요한 모든 것이 포함되어 있다.

#### YARN에서의 스파크

* 스파크의 가장 일반적인 배포 방법은 하둡과 함께 제공되는 YARN 리소스 관리 프레임워크를 사용하는 것이다.
* YARN은 하둡 클러스터에서 workload를 예약하고 관리할 수 있는 하둡의 핵심 구성 요소이다.
* Databricks 연례 조사에 따르면 YARN과 독립실행형은 거의 같고, 메소스는 그보다 뒤에 있다. (사용 빈도수)
* 하둡 에코시스템의 일급 객체(first-class citizens)답게 스파크 응용 프로그램은 최소한의 노력으로 제출하고 관리를 쉽게 할 수 있다.
* Driver, Master, Executor 와 같은 스파크 프로세스는 리소스 매니저, 노드 매니저, 어플리케이션 마스터와 같이 YARN 프로세스에 의해 호스팅되거나 촉진된다.
* spark-submit, pyspark, spark-shell 프로그램에는 스파크 응용 프로그램을 YARN 클러스터에 제출하는 데 사용되는 커맨드 라인 인수가 포함된다. (다음 코드 참조)
* YARN을 스케줄러로 사용할 때 cluster와 client라는 2개의 클러스터 배포 모드가 있다.

```console
# YARN 클러스터에 스파크 작업 제출

$SPARK_HOME/bin/spark-submit \
--class org.apache.spark.examples.SparkPi \
--master yarn \
--deploy-mode cluster \

&SPARK_HOME/examples/jars/spark-examples*.jar 10
```

#### 메소스에서의 스파크

* 아파치 메소스는 버클리 캘리포니아 대학에서 개발한 오픈소스 클러스터 매니저이다. 
* 스파크의 생성을 포함한 lineage의 일부를 공유한다.
* 여러 유형의 응용 프로그램을 스케줄링할 수 있다.
* 클러스터의 활용도를 높이기 위해 세분화된 리소스 공유 기능을 제공한다.
* 다음 코드는 메소스 클러스터에 제출된 스파크 응용 프로그램의 예제이다.
* 이 책은 스파크 독립실행형 및 YARN에 대한 일반적인 스케줄러에 중점을 둔다. (메소스에 대한 자세한 설명: https://mesos.apache.org)

```console
# 메소스 클러스터에 스파크 작업 제출

$SPARK_HOME/bin/spark-submit \
--class org.apache.spark.examples.SparkPi \
--master mesos://mesosdispatcher:7077 \
--deploy-mode cluster \
--supervise \
--executor-memory 20G \
--total-executor-cores 100 \

&SPARK_HOME/examples/jars/spark-examples*.jar 1000
```

### 2.2. 스파크 설치 준비

...

### 2.3. 스파크 설치 탐색

* 스파크 설치 디렉토리(SPARK_HOME)의 내용을 숙지하는 것이 좋다. 
   * `bin/` - pyspark, spark-shell, spark-sql 및 sparkR과 같은 shell 프로그램을 통해, 또는 spark-submit을 사용하는 일괄 처리 모드에서 대화식으로 스파크 응용 프로그램을 실행하는 모든 명령/스크립트를 포함한다.
   * `conf/`
   * `data/`
   * `examples/`
   * `jars/`
   * `licenses/`
   * `python/`
   * `R/`
   * `sbin/`
   * `yarn/`

### 2.4. 다중노드(Multi-Node) 스파크 독립실행형 클러스터 배포

...

### 2.5. 클라우드에서 스파크 배포

* SaaS(Software as a Service), IaaS(Infrastructure as a Service), PaaS(Platform as a Service)와 같은 공공 및 사설 클라우드 기술의 확산은 조직이 기술을 배포하는 방식의 판도를 바꿨다.
* 스파크를 클라우드에서 배포하면 빠르고 확장 가능하며, 탄력적인 프로세스 환경을 제공할 수 있다.

...

[[top](#contents)]

<br>

<div id='3.'/>

## 3. 스파크 클러스터 아키텍처의 이해

### 3.1. 스파크 응용 프로그램의 해부

* 단일 기기든 수백 수천 개의 노드로 구성된 클러스터든 스파를 실행하는 응용 프로그램은 모두 구성 요소가 있다.
* 각 구성 요소는 스파크 응용 프로그램을 실행하는 동안 필요하며, 특정한 역할이 있다.
* client와 같은 일부 구성 요소는 실행 중에 수동적으로 작동하지만, 계산 함수를 포함한 또 다른 구성 요소는 프로그램 실행 중에 활성화된다.
* 구성 요소는 Driver, Master, Cluster Manager, 작업자 노드를 실행시키는 Executor(실행자), 그리고 Workers(작업자)이다.
* 다음 그림은 스파크 독립실행형 응용 프로그램 콘텍스트의 모든 스파크 구성 요소를 보여준다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_3_1.png" width="60%" height="60%"></p>

* Driver, Master, Executor 프로세스를 포함한 모든 스파크 구성 요소들은 JVM(Java Virtual Machine)에서 실행된다. 
* JVM은 자바 바이트코드로 컴파일된 명령어를 실행할 수 있는 크로스 플랫폼 런타임 엔진이다.
* 스파크로 작성된 Scala는 바이트코드로 컴파일되고 JVM에서 실행된다.
* 스파크의 런타임 응용 프로그램 구성 요소와 실행되는 위치 및 노드의 유형을 구별할 수 있어야 한다.
* 이러한 구성 요소들은 다양한 배포 모드를 사용하며, 모두 다른 위치에서 실행되기 때문에 이들을 물리적 노드나 인스턴스 용어로 생각하면 안된다.
   * 예를 들어, YARN에서 스파크를 실행하면 몇 가지 변형은 생기지만, 생성된 모든 구성 요소는 여전히 응용 프로그램에 포함되어 동일한 역할을 수행한다.

#### 스파크 Driver

* 스파크 응용 프로그램의 수명은 스파크 Driver에서 시작하고 끝난다. 
* Driver는 client가 스파크에서 응용 프로그램을 제출하는 데 사용하는 프로세스이다.
* Driver는 스파크 프로그램의 실행을 계획 및 조정하고, 상태 및 결과(데이터)를 client에게 반환한다.
* Driver는 클러스터에 있는 client나 노드에 물리적으로 상주할 수 있다.

##### SparkSession

* 스파크 Driver는 SparkSession을 생성한다.
* SparkSession 객체는 스파크 클러스터에 대한 연결을 나타낸다.
* SparkSession은 대화식 shell을 포함해 스파크 응용 프로그램의 시작 부분에서 인스턴스화되며, 프로그램 전체에서 사용된다.
* 스파크 2.0 이전에는 스파크 핵심 응용 프로그램에 사용된 `SparkContext`, 스파크 SQL 응용 프로그램과 함께 사용된 `SQLContext`, `HiveContext`, 스파크 스트리밍 응용 프로그램에 사용된 `StreamingContext`가 스파크 응용 프로그램의 entry point에 포함되어 있었다. 스파크 2.0의 SparkSession 객체는 이러한 모든 객체를 단일 entry point로 결합하여 모든 스파크 응용 프로그램에 사용할 수 있다.
* SparkSession 객체는 SparkContext, SparkConf 자식 객체를 통해 Master, 응용 프로그램 이름, Executors의 개수 등 사용자가 설정한 모든 런타임 구성 속성을 포함한다.
* 다음 그림은 pyspark shell 내의 SparkSession 객체와 그 구성 속성 중 일부를 보여준다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_3_2.png" width="60%" height="60%"></p>

* 다음 코드는 spark-submit으로 제출된 프로그램과 같은 비대화식 스파크 응용 프로그램 내에서 SparkSession을 생성하는 방법을 보여준다.

```python
# SparkSession 생성하기

from pyspark.sql import SparkSession
spark = SparkSession.builder \
	.master("spark://sparkmaster:7077") \
	.appName("My Spark Application") |
	.config("spark.submit.deployMode", "client") \
	.getOrCreate()

numlines = spark.sparkContext.textFile("file:///opt/spark/licenses") \
	.count()

print("The total number of lines is " + str(numlines))
```

##### 응용 프로그램 계획

* Driver의 주요 기능 중 하나는 응용 프로그램을 계획하는 것이다.
* Driver는 응용 프로그램 프로세싱 입력에 따라 프로그램을 어떻게 실행할지 계획한다. 
* Driver는 요청된 모든 transformation 및 action에 따라 노드의 DAG(Directed Acyclic Graph)를 작성한다. 
   * 여기서 transformation은 데이터 조작 작업을, action은 출력 또는 prompt에 대한 프로그램 실행 요청을, 각 노드는 변형 또는 계산 단계를 나타낸다.
* DAG(지시된 비순환 그래프)란? 
   * DAG는 데이터 흐름(data flow)과 그 종속성을 나타내기 위해 컴퓨터 과학에서 일반적으로 사용되는 수학적 구조이다.
   * DAG는 정점(vertice), 노드 및 에지(edge)로 구성된다.
   * 데이터 흐픔 콘텍스트의 정점은 프로세스 흐름의 단계이다.
   * DAG의 에지는 정점을 서로 연결하는데, 순환 참조를 가질 수 없는 방식으로 연결한다.
* 스파크 응용 프로그램 DAG는 task와 stage로 구성된다. 
   * task는 스파크 프로그램에서 스케줄링이 가능한 일의 최소 단위
   * stage는 함께 실행할 수 있는 일련의 작업 모음
   * stage는 서로 의존하는 stage dependencies(단계 종속성)이 있다.
* 프로세스 스케줄링에서 DAG는 스파크에만 고유하게 있는 것은 아니다. Tez, Drill, Presto와 같은 다른 빅데이터 에코시스템 프로젝트에서도 스케줄링을 위해 사용한다.
* DAG는 스파크의 기본 요소이므로 개념을 잘 이해해야 한다.

##### 응용 프로그램 조직화(Orchestration)

* Driver는 DAG에 정의된 stage 및 task의 실행을 설계한다.
* 스케줄링 및 task 실행과 관련된 Driver의 주요 활동
   * task 실행에 사용할 수 있는 리소스 추적
   * 가능한 데이터에 'close'를 실행하는 작업 스케줄링 (데이터 지역성 개념)

##### 기타 함수

* Driver는 스파크 프로그램의 실행을 계획하고 조정하는 것 외에도 응용 프로그램의 결과를 반환하는 책임이 있다.
* 데이터를 client에 반환하도록 요청되는 action의 경우 반환값은 리턴 코드 또는 데이터가 될 수 있다.

#### 스파크 Executor 및 Worker

* 스파크 Executor는 스파크 DAG 작업이 실행되는 프로세스이다.
* Executor는 스파크 크러스터에 slave 노드 또는 worker의 CPU 및 메모리 리소스를 예약한다. 이는 특정 스파크 응용 프로그램 전용이며, 응용 프로그램이 완료되면 종료된다.
* 스파크 프로그램은 보통 많은 Executor로 구성되며, 종종 병렬로 작업한다.
* 일반적으로 Executor 프로세스를 호스팅하는 Worker 노드에는 유한하거나 고정된 수의 Executor가 특정 시점에 할당된다. 
   * 따라서, 노드의 수를 알고 있는 클러스터에는 주어진 시간에 실행할 수 있는 Executor의 수가 한정된다.
   * 응용 프로그램이 클러스터의 실제 용량을 초과해 Executor를 요구하면, 다른 Executor가 리소스를 완료하고 릴리스하는 것으로 시작되도록 예약한다.
* 스파크 Executor를 호스트하는, Executor용 JVM에는 객체를 저장하고 관리하는 전용 메모리 공간인 heap이 할당된다.
* Executor의 JVM heap에 커밋된 메모리 양은 spark.executor.memory 속성 또는 pyspark.spark-shell 이나 spark-submit 명령에 대한 --executor-memory 인수로 설정된다.
* Executor는 task의 출력 데이터를 메모리 또는 디스크에 저장한다.
* Worker와 Executor는 할당된 task에만 의식하지만, Driver는 응용 프로그램을 구성하는 전체 작업 집합과 관련된 종속성을 파악해야 한다.

#### 스파크 Master와 Cluster Manager

* 스파크 Driver는 스파크 응용 프로그램을 실행하는 데 필요한 일련의 작업을 계획하고 조정한다. Worker 노드에 호스트되는 Executor에서 task가 실행된다.
* Master 및 Cluster Manager는 Executor가 실행되는 분산 클러스터 리소스(YARN이나 메소스의 경우 컨테이너)를 모니터링하고 예약하고 할당하는 중앙 프로세스이다.
* Master와 Cluster Manager는 독립된 프로세스로 서로 분리될 수 있고, 독립실행형 모드에서 스파크를 실행할 때처럼 하나의 프로세스로 결합될 수도 있다.

##### 스파크 Master

* 스파크 Master는 클러스터의 리소스를 요청하고 이를 스파크 Driver에서 사용할 수 있게 만드는 프로세스이다.
* 모든 배포 모드에서, Master는 worker 노드 또는 slave 노드에 리소스나 컨테이너를 할당하고, 그 상태를 추적하고 진행 상황을 모니터링한다.
* 스파크 독립실행형 모드로 실행하면, 스파크 Master 프로세스는 마스터 호스트의 포트 8080에서 웹 UI를 제공한다.
* 스파크 Master vs. 스파크 Driver
   * Driver와 Master의 런타임 함수를 구별하는 것은 매우 중요하다.
   * Master라는 명칭은 프로세스가 응용 프로그램의 실행을 통제한다는 의미로 추론할 수 있지만, 실제로 그렇지 않다.
   * Master는 단순히 리소스를 요청해서 Driver가 사용할 수 있게 한다.
   * Master는 이렇게 리소스의 위치와 상태는 모니터링하지만, 응용 프로그램의 실행 및 해당 작업과 단계의 조정에는 관여하지 않는다. 그것은 Driver의 역할이다.

##### Cluster Manager

* Cluster Manager는 worker 노드를 모니터링하고, Master가 요청하면 이러한 노드의 리소스를 예약하는 프로세스이다.
* Master는 이렇게 클러스터가 예약한 리소스를 Executor의 형태로 Driver에 제공한다.
* 메소스나 YARN에서 스파크를 실행할 때, cluster manager는 master 프로세스와 분린된다.
* 독립실행형 모드로 실행되는 스파크의 경우, master 프로세스는 cluster manager의 함수도 수행하고, cluster manager 역할도 수행한다.
* cluster manager 함수는 하둡 클러스터에서 실행되는 스파크 응용 프로그램용 YARN 리소스 매니저 프로세스로 적합하다.
* 리소스 매니저는 YARN 노드 매니저에서 실행되는 컨테이너의 상태를 예약, 할당 및 모니터링한다. 스파크 응용 프로그램은 clutser 모드에서 실행 중인 응용 프로그램의 master 프로세스처럼 이 컨테이너를 사용해 executor 프로세스를 호스트한다.

### 3.2. 독립실행형 스케줄러를 사용하는 스파크 응용 프로그램

...

### 3.3. YARN에서 실행되는 스파크 응용 프로그램의 배포 모드

#### Client 모드

* Client 모드에서 Driver 프로세스는 응용 프로그램을 제출하는 client에서 실행된다.
* 기본적으로 관리되지 않으므로 Driver 호스트가 실패하면 그 응용 프로그램도 실패한다.
* Client 모드는 대화식 셸 세션(pyspark, spark-shell)과 비대화식 응용 프로그램 제출(spark-submit) 모두 지원한다.
* 다움 코드는 Client 배포 모드를 사용해 pyspark 세션을 시작하는 방법을 보여 준다.

```
# 코드 3.2. YARN Client 배포 모드

$SPARK_HOME/bin/pyspark \
--master yarn-client \
--num-executors 1 \
--driver-memory 512m \
--executor-memory 512m \
--executor-cores 1

# OR

$SPARK_HOME/bin/pyspark \
--master yarn \
--deploy-mode client \
--num-executors 1 \
--driver-memory 512m \
--executor-memory 512m \
--executor-cores 1
```

* 다음 그림은 client 모드로 YARN에서 실행되는 스파크 응용 프로그램의 개요를 보여 준다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_3_3.png" width="60%" height="60%"></p>

1. 클라이언트는 스파크 응용 프로그램을 클러스터 매니저(YARN 리소스 매니저)에 제출한다. Driver 프로세스, SparkSession 및 SparkContext가 만들어지고, 클라이언트에서 실행된다.
2. 리소스 매니저는 응용 프로그램에 대해 애플리케이션 마스터(스파크 마스터)를 할당한다.
3. 애플리케이션 마스터는 리소스 매니저에게 실행자로 사용될 컨테이너를 요청한다. 할당된 컨테이너로 실행자가 생성된다.
4. 클라이언트에 속한 Driver는 스파크 프로그램의 작업 및 단계 프로세스를 감시하기 위해 실행자와 통신한다. Driver는 진행률, 결과 및 상태를 클라이언트에 보고한다.
<br>

* Client 배포 모드는 사용하기에 가장 간단한 모드이지만, 대부분의 프로덕션 응용 프로그램에 필요한 복원력은 부족하다.

#### Cluster 모드

* Client 모드 배포 모드와 달리, YARN 클러스터 모드에서 실행되는 스파크 응용 프로그램과 Driver는 애플리케이션 마스터의 하위 프로세스로 클러스터에서 실행된다.
* 만약 Driver를 호스팅하는 애플리케이션 마스터 프로세스가 실패하면, 클러스터의 다른 노드에서 다시 인스턴스화될 수 있다. 이처럼 Cluster 모드는 탄력적이다.
* Driver가 클러스터에서 실행 중인 비동기 프로세스이므로 Cluster 모드는 대화형 셸 응용 프로그램(pyspark 및 spark-shell)에서 지원되지 않는다.
* 다음 코드는 spark-submit과 YARN 클러스터 배포 모드를 사용해 응용 프로그램을 제출하는 방법을 보여 준다.

```
# 코드 3.3. YARN Cluster 배포 모드

$SPARK_HOME/bin/spark-submit \
--master yarn-cluster \
--num-executors 1 \
--driver-memory 512m \
--executor-memory 512m \
--executor-cores 1
$SPARK_HOME/examples/src/main/python/pi.py 10000

# OR

$SPARK_HOME/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--num-executors 1 \
--driver-memory 512m \
--executor-memory 512m \
--executor-cores 1
$SPARK_HOME/examples/src/main/python/pi.py 10000
```

* 다음 그림은 Cluster 모드의 YARN에서 실행되는 스파크 응용 프로그램의 개요를 나타낸다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_3_4.png" width="60%" height="60%"></p>

1. 클라이언트(spark-submit을 호출하는 사용자 프로세스)는 스파크 응용 프로그램을 클러스터 매니저(YARN 리소스 매니저)에 제출한다.
2. 리소스 매니저는 응용 프로그램에 적합한 애플리케이션 마스터(스파크 마스터)를 할당한다. 그러면 동일한 클러스터 노드에서 Driver 프로세스가 생성된다.
3. 애플리케이션 마스터는 리소스 매니저에게 실행자용 컨테이너를 요청한다.
   * 실행자는 리소스 매니저에 의해 애플리케이션 마스터에 할당된 컨테이너 안에 생성된다.
   * 그런 다음 Driver는 스파크 프로그램의 작업 및 단계 프로세스를 감시하기 위해 실행자와 통신한다.
4. 클러스터 노드에서 실행 중인 Driver는 진행률, 결과 및 상태를 클라이언트에 보고한다.

* 이와 같이 스파크 응용 프로그램 웹 UI는 클러스터의 애플리케이션 마스터 호스트에서 사용할 수 있다.
* 이 사용자 인터페이스에 대한 링크는 YARN 리소스 매니저 UI를 통해 이용할 수 있다.

...

[[top](#contents)]

<br>

<div id='4.'/>

## 4. 스파크 프로그래밍 기초 학습

### 4.1. RDD의 소개

* RDD(Resilient Distributed Dataset)은 스파크 프로그래밍에서 사용되는 가장 기본적인 데이터 객체이다.
* RDD는 스파크 응용 프로그램 내의 데이터 집합으로, 로드된 초기 데이터 집합, 중간 데이터 집합 및 최종 결과 데이터 집합을 모두 포함한다.
* 대부분의 스파크 응용 프로그램은 외부 데이터로 RDD를 로드해 연산을 수행한 뒤 새로운 RDD를 생성하는데, 이러한 작업을 변환(transformation)이라 한다.
* transformation 프로세스는 궁극적으로 원하는 결과물이 출력될 때까지 반복된다. 원하는 결과물을 출력하는 작업의 유형을 action이라 한다.
   * 응용 프로그램의 결과를 파일 시스템에 기록하는 것 등을 액션의 예로 들 수 있다.
<br>

* RDD는 분산된 객체 모음이다. 여기서 객체는 스파크 프로그램에서 사용되는 데이터를 나타낸다.
   * PySpark의 RDD는 list, tuple, dictionary와 같은 분산 파이썬 객체로 구성된다.
   * RDD 내의 객체는 list의 요소와 같이 integer, floating, string과 같은 기본 데이터 유형은 물론, tuple, list, dictionary와 같은 복합 유형을 포함한 모든 유형이 된다.
   * 스칼라 및 자바 API에서 RDD는 각각 스칼라 또는 자바 객체 모음으로 구성된다.
* RDD를 디스크에 지속시키기 위한 옵션이 있지만, RDD는 주로 메모리에 저장되거나 적어도 가능한 한 메모리에 저장되도록 한다.
* 스파크의 초기 용도는 머신러닝을 지원하는 것이었으므로, 스파크 RDD는 제한된 형식의 공유 메모리를 제공한다. 이는 연속적이고 반복적인 연산을 수행할 때, 데이터의 효율적인 재사용을 가능하게 한다.
<br>

* 하둡 맵리듀스 구현의 주요 단점 중 하나는 중간 데이터를 디스크에 지속적으로 저장하고, 런타임 중 노드 간에 데이터를 복사하는 것이었다.
* 이러한 맵리듀스의 데이터 공유 분산 처리 방식은 복원력과 내결함성을 제공하지만, 지연 시간이 발생했다. 
* 데이터 양이 증가함에 따라 실시간 데이터 처리 및 통찰력이 필요해지고, 이로 인해 스파크의 RDD에 기반을 둔 인-메모리 처리 프레임워크는 인기를 끌었다.
<br>

* 복원 분산 데이터 집합(RDD, Resilient Distributed Dataset)이라는 용어는 그 자체로 개념이 정확하고 간결하다.
   * 복원(Resilient) - RDD는 탄력적이다. 
      * 스파크에서 작업을 수행하다가 노드가 손실될 경우 데이터 집합을 재구성할 수 있다.
      * 이것은 스파크가 각 RDD 리니지(RDD를 만드는 일련의 단체)를 알고 있기에 가능하다.
   * 분산(Distributed) - RDD는 분산된다.
      * 파티션을 하나 또는 여러 개로 나누고, 클러스터의 작업자 노드를 통해 RDD의 데이터를 메모리 내 객체의 모음으로 분산한다.
      * RDD는 서로 다른 노드(작업자)의 프로세스(실행자) 사이에 데이터를 교환하기 위한 효율적인 공유 메모리 형태를 제공한다.
   * 데이터 집합(Dataset) - RDD는 레코드로 구성된 데이터 집합이다.
      * 레코드는 데이터 집합 안의 고유하고 식별 가능한 데이터 모음으로, 관계형 데이터베이스의 테이블에 있는 행이나 파일 내 텍스트의 줄 또는 여러 다른 형식의 필드와 유사한 필드 모음을 말한다.
      * RDD는 각 파티션이 고유한 레코드 세트를 포함하도록 작성되며, 독립적으로 동작할 수 있다. 이것이 비공유(shared nothing) 접근법이다.
<br>

* RDD의 또 다른 주요 특성은 '불변성'이다.
   * RDD가 일단 인스턴스화되고 데이터로 채워진 후에는 업데이트할 수 없다는 것을 의미한다.
   * 대신에, map 또는 filter 함수와 같은 변환(tranformations) 연산을 통해 기존 RDD를 기반으로 새로운 RDD를 만들 수는 있다.
<br>

* 액션은 RDD에서 수행되는 다른 연산이다.
   * 액션은 드라이버 프로그램으로 반환된 RDD에서 데이터 형식으로 존재할 수 있는 출력을 만들거나, RDD의 콘텐츠를 파일 시스템(로컬, HDFS, S3, ...)에 저장한다.
   * 이 밖에도 RDD 내의 레코드 수를 반환하는 등 많은 다른 액션이 있다.
<br>

* 다음 코드는 RDD에 데이터를 로드하고 filter 변환을 사용해 새로운 RDD를 생성한 다음, 액션을 사용해 결과 RDD를 디스크에 저장하는 스파크 프로그램 샘플을 보여준다.

```python
# 코드 4.1. 로그 파일에서 오류를 검색하는 pyspark 프로그램 샘플

# 로컬 파일 시스템에서 로그 파일 로드
logfilesrdd = sc.textFile("file:///var/log/hadoop/hdfs/hadoop-hdfs-*")
# 오류에 대해서만 로그 레코드 필터링
onlyerrorsrdd = logfilesrdd.filter(lambda line: "ERROR" in line)
# onlyerrorsrdd를 파일로 저장
onlyerrorsrdd.saveAsTextFile("file:///tmp/onlyerrorsrdd")
```

* RDD 개념에 대한 자세한 내용은 Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing 논문을 살펴봐라.


### 4.2. RDD에 데이터 로드하기

...

#### 데이터 소스에서 RDD 만들기

* 데이터베이스에서 스파크 프로그램의 RDD로 데이터를 로드하기 위해서 historical 데이터, master 데이터, reference 데이터, lookup 데이터의 소스가 필요하다.
   * 이 데이터는 오라클, MySQL, Postgres 및 SQL 서버를 포함한 다양한 호스트 시스템 및 데이터베이스 플랫폼에서 가져올 수 있다.
* 외부 파일을 사용해서 만든 RDD처럼 외부 데이터베이스의 데이터를 사용해 생성된 RDD는 여러 작업자의 여러 partition으로 데이터를 이동하려고 시도한다.
   * 이는 처리 중, 특히 초기 단계에서 병렬 처리를 극대화한다.
   * 또한, 키 공간을 기준으로 테이블을 다른 partition으로 나눌 경우 partition이 병렬로 로드될 수 있으며, 이때 각 parition은 고유한 행 세트를 가져와야 한다. (다음 그림 참조)

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_4_1.png" width="60%" height="60%"></p>

* 관계형 데이터베이스 table 또는 query 에서 RDD를 생성하는 기본 방법은 SparkSession 객체의 함수를 사용하는 것이다. 이것이 스파크 내 모든 유형의 데이터 작업을 위한 주요 entry point이다.
* SparkSession은 DataFrameReader 객체를 반환하는 read 함수를 제공한다. (SchemaRDD라고 불렸던 RDD의 특수 유형인 DataFrame으로 데이터를 읽기 위해 이 객체를 사용)
* read() 메소드에는 JDBC 호환 데이터 소스에 연결해 데이터를 수집할 수 있는 jdbc 함수가 있다.

```console
# pyspark 실행 및 JDBC MySQL 커넥터 JAR 파일 제공

## 대상 데이터베이스에 대한 최신 jdbc 커넥터 다운로드
$SPARK_HOME/bin/pyspark \
--driver-class-path mysql-connector-java-5.*-bin.jar \
--master local
```

* employees라는 테이블, employees라는 데이터베이스, mysqlserver라는 이름의 MySQL 서버가 있다.
* employees 테이블에는 emp_no라는 기본키가 있는데, 이 키는 논리적으로 테이블의 키 공간을 여러 partition으로 나누는 데 사용한다.
* JDBC를 통해 MySQL 데이터베이스에 접근하려면 드라이버 클래스 경로에 mysql-connector.jar를 제공하는 pyspark를 실행해야 한다.
* mysql-connector.jar와 같은 커넥터는 일반적으로 대상 데이터베이스 플랫폼 공급 업체의 웹 사이트에서 사용할 수 있다. (위 코드 참조)
* 대상 데이터베이스에 대한 관련 JDBC 연결 라이브러리를 포함해 대화식 또는 비대화식 스파크 응용 프로그램을 실행하면, DataFrame Reader 객체의 jdbc 메소를 사용할 수 있다.
* 다음 코드는 read.jdbc() 메소드를 사용해 RDD를 생성하는 방법을 보여준다.

```python
employeesdf = spark.read.jdbc(
   url="jdbc:mysql://localhost:3306/employees", 
   table="employees", 
   column="emp_no", 
   lowerBound="10001", 
   upperBound="499999", 
   numParition="2", 
   #predicates=None, 
   properties={"user":"<user>", "password":"<pwd>"})

employeesdf.rdd.getNumPartitions()
# numParitions=2 이므로 2를 반환한다.
```

* url 및 table 인수는 읽을 대상 데이터베이스와 테이블을 지정한다.
* column 인수는 스파크가 numPartitions 에서 지정한 partition 수를 생성하기 위해 적절한 열(선호하는 long 또는 int 데이터 유형)을 선택하는 데 도움이 된다.
* upperBound와 lowerBound 인수는 스파크의 partition 생성을 돕기 위해 column 인수와 함께 사용된다. 이들은 소스 테이블의 지정된 열에 대한 최솟값과 최댓값을 나타낸다. 이 인수 중 하나라도 read.jdbc() 함수와 함께 제공되면, 모두 값이 정해져야 한다.
* 선택적 인수 predicates는 파티션을 로드하는 동안 불필요한 레코드를 필터링하기 위해 WHERE 조건을 포함시킨다.
* properties 인수를 사용해 매개변수를 JDBC API에 전달할 수 있다. 이 인수가 제공되려면 다양한 구성 옵션을 나타내는 일련의 이름/값 쌍이 딕셔너리로 표현되어야 한다.
* read.jdbc() 함수는 다음 코드와 같이 DataFrame을 반환한다.

```python
sqlContext.registerDataFrameAsTable(employeesdf, "employees")

df2 = spark.sql("SELECT emp_no, first_nmae, last_name FROM employees LIMIT 2")
df2.show()

#>>>
#+------+----------+---------+
#|emp_no|first_name|last_name|
#+------+----------+---------+
#| 10001|   Georigi|  Facello|
#| 10002|   Bezalel|   Simmel|
#+------+----------+---------+
```

* read.jdbc() 함수를 사용해 너무 많은 파이션을 생성하는 경우
   * 관계형 데이터 소스에서 데이터프레임을 로드할 때 너무 많은 파이션을 지정하지 않도록 주의해야 한다.
   * 각 개별 작업자 노드에서 실행되는 각 파티션은 DBMS에 독립적으로 연결되고, 데이터세트의 지정된 부분을 쿼리한다.
   * 수백 또는 수천 개의 파티션이 있는 경우 이는 호스트 데이터베이스 시스템에 대한 분산 서비스 거부(DDoS, Distributed denial-of-service) 공격으로 잘못 해석될 수 있다.

...


### 4.3. RDD 연산

#### 주요 RDD 개념

* 스파크의 trnaformation은 RDD에서 작동하고 새로운 RDD를 반환하는 함수인 반면, action은 RDD에 대한 연산 작업 후 값을 반환하거나 출력 연산을 수행하는 함수다.

##### 거친(Coarce-Grained) transformation vs 세분화된(Fine-Grained) transformation

* RDD에 대해 수행되는 연산은 데이터 집합의 모든 요소에 대해 함수(map(), filter())를 적용하고, transformation이 적용된 새 데이터 집합을 반환하므로 '거친' 것으로 간주된다.
* '거친' transformation과 달리 세분화된 transformation은 관계형 데이터베이스의 단일 행 업데이트 또는 NoSQL 데이터베이스의 put 연산과 같이 단일 레코드나 데이터 셀을 조작할 수 있다.
* 거친 transformation은 하둡의 맵리듀스 프로그래밍 모델 구현과 개념적으로 비슷하다.

##### Transformations, Actions, and Lazy Evaluation

* Transformation은 RDD에 대해 수행되는 연산으로 새로운 RDD를 생성한다. 
* 일반적인 transformation에는 map(), filer() 함수가 있다.

```python
originalrdd = sc.parallelize([0, 1, 2, 3, 4, 5, 6, 7, 8])
newrdd = originalrdd.fileter(lambda x: x % 2)
```

* originalrdd는 수의 병렬화된 모음에서 시작한다.
* filter() transformation을 originalrdd의 각 요소에 적용해 컬렉션의 짝수를 건너뛴다. 이 결과 newrdd라는 RDD가 생성된다.
* 새로운 RDD 객체를 반환하는 transformation과 달리 action은 값 또는 데이터를 Driver 프로그램에 반환한다.
* 일반적인 action에는 reduce(), collect(), count(), saveAsTextFile()이 있다.

```python
newrdd.collect() # will return [1, 3, 5, 7]
```

* 스파크는 스파크 프로그램을 프로세싱할 때 lazy evaluation을 사용한다.
* Lazy evaluation은 lazy execution이라고도 하며, action이 호출될 때까지(즉, 출력이 필요할 때) 프로세스를 연기한다.
* 이는 대화형 shell을 사용해 쉽게 나타낼 수 있다.
* 대화형 shell에서는 어떤 프로세스도 시작하지 않고 RDD에 대해 하나 이상의 transformation 방법을 차례로 입력할 수 있다.
* 대신 각 문장은 구문과 객체 참조에 대해서만 구문 분석이 된다.
* count()나 saveAsTextFile()과 같은 action을 요청하면, 논리적이고 물리적인 실행 계획에 따라 DAG가 만들어진다. Driver는 Executors 간에 이렇나 계획을 조정하고 관리한다.
* 이러한 lazy evaluation을 통해 스파크는 가능한 한 작업을 결합해 처리 단계를 줄이고, shuffling이라는 프로세스에서 스파크 executor 사이에 전송되는 데이터의 양을 최소화한다.

##### RDD 지속성 및 재사용

* RDD는 주로 실행자(Executors)의 메모리에 만들어지고 존재한다.
* RDD는 기본적으로 필요할 때만 존재하는 일시적인 객체다.
* 새로운 RDD로 변환되고 더 이상 어떤 연산에도 필요하지 않게 되면 영구적으로 제거된다.
* RDD는 매번 다시 전체적으로 평가되므로 하나 이상의 액션에 RDD가 필요한 경우에 문제가 될 수 있다.
* 이 문제를 해결하기 위한 방법은 `.persist()` 메소드를 사용해 RDD를 캐시하거나 유지하는 것이다.

```python
# 코드 4.11. 지속성이 없는 다중 액션에 RDD 사용하기

numbers = sc.range(0, 1000000, 1, 2)
evens = numbers.filter(lambda x: x % 2)
noelements = evens.count()
# evens RDD 처리
print("There are %s elements in the collection" % (noelements))
# return "There are 500000 elements in the collection"
listofelements = evens.collect()
# evens RDD 재처리
print("The first five elements include " + (str(listofelements[0:5])))
# return "The first five elements include [1, 3, 5, 7, 9]"
```

```python
# 코드 4.12. 지속성이 있는 다중 액션에 RDD 사용하기

numbers = sc.range(0, 1000000, 1, 2)
evens = numbers.filter(lambda x: x % 2)
evens.persist()
# 다음 액션이 요청될 때마다 evens RDD를 계속 유지하도록 스파크에 지시한다.
noelements = evens.count()
# 메모리에서 evens RDD를 처리하고 지속한다.
print("There are %s elements in the collection" % (noelements))
# return "There are 500000 elements in the collection"
listofelements = evens.collect()
# evens RDD를 재계산할 필요가 없다.
print("The first five elements include " + (str(listofelements[0:5])))
# return "The first five elements include [1, 3, 5, 7, 9]"
```

* persist() 메소드를 사용해 RDD를 유지시키면, 첫 번째 액션이 호출된 후 계산된 클러스터의 모든 노드 안에 있는 각 메모리에 RDD가 남아 있는 것을 확인할 수 있다.
* cache() 메소드가 이와 비슷한 기능을 한다.
* 스파크 응용 프로그램 UI의 저장소(storage) 탭을 확인하면 RDD가 남아 있는 것을 알 수 있다.

...

#### 맵리듀스 및 Word Count 연습

* 맵리듀스는 대부분의 빅데이터 및 NoSQL 플랫폼의 중심에 있는 플랫폼 및 언어 독립적 프로그래밍 모델 또는 디자인 패턴이다.
* Map 또는 Reduce 함수를 명시적으로 구현하지 않고 데이터를 처리할 수 있는 Pig나 Hive와 같이 맵리듀스에는 많은 추상화가 존재한다.
* 맵리듀스의 개념을 이해하는 것은 스파크에서의 분산 프로그래밍 및 데이터 처리를 진정으로 이해하는 데 필수적이다.

```
# 1. 단일 노드(single-node) 스파크 설치를 사용해 다음 링크에서 shakespeare.txt 파일을 다운로드한다. (wget or curl)
   https://s3.amazonaws.com/sparkusingpython

# 2. 스파크 설치의 /opt/spark/data 디렉토리에 파일을 저장한다.
   $ sudo mv shakespeare.txt /opt/spark/data

   사용 가능한 HDFS가 있는 경우 파일을 HDFS에 업로드하고 대체 파일로 사용할 수 있다.

# 3. 로컬 모드에서 pyspark shell을 연다.
   $ pyspark --master local

   하둡 클러스터 또는 분산된 스파크 독립실행형 클러스터에 접근할 수 있는 경우, 다음 중 하나를 지정해 자유롭게 사용할 수 있다.
   --master yarn
   --master spark://<yoursparkmaster>:7077

   파이썬 바이너리가 python이 아니라면, 스파크로 올바른 파일로 보내야 한다. 이 작업은 다음 환경변수 설정을 사용한다.
   $ export PYSPARK_PYTHON=python3
   $ export PYSPARK_DIVER_PYTHON=python3
```

```python
# 4.
import re

# 5.
doc = sc.textFile("file:///opt/spark/data/shakespeare.txt")

# 6. 빈 줄은 필터링, 공백으로 줄을 나누고, 단어 목록을 하나의 목록으로 평평하게 만든다.
flattened = doc \
   .filter(lambda line: len(line) > 0) \
   .flatMap(lambda line: re.split('\W+', line))

# 7.
flattened.take(6)

# 8. 텍스트를 소문자로 매핑하고 빈 문자열을 제거한 후, 다음 양식(word, 1)의 키/값 쌍으로 변환한다.
kvpairs = flattened \
   .filter(lambda word: len(word) > 0) \
   .map(lambda word: (word.lower(), 1))

# 9.
kvpairs.take(5)

# 10. 각 단어를 세고, 결과를 역알파벳 순서로 정렬한다.
countsbyword = kvpairs \
   .reduceByKey(lambda v1, v2: v1 + v2) \
   .sortByKey(ascending=False)

# 11.
countsbyword.take(5)

# 12. 가장 많이 사용된 상위 5개 단어를 찾는다.
# 카운트를 키와 정렬로 바꾸러면 kv 쌍을 반전한다.
topwords = countsbyword \
   .map(lambda x: (x[1], x[0]))
   .sortByKey(ascending=False)

# 13. 
topwords.take(5)
```
```
map() 함수는 키와 값을 반전하는 12단계에서 사용되는 방법을 참고하면 된다.
이는 기본적으로 정렬되지 않은 값을 정렬하는 수단인 secondary sort라는 연산을 수행하는 일반적인 방법이다.

이제 Ctrl + D 를 눌러서 pyspark 세션을 종료한다.
```

```
# 14. 이제 모든 것을 하나로 합친 다음 spark-submit을 사용해 완전한 파이썬 프로그램으로 실행한다.
   먼저, 스파크 설치의 conf 디렉토리에 log4j.properties 파일을 생성하고 설정함으로써 로깅의 양을 최소화한다. (다음 명령어 참고)

# minimize_logging.sh
sed "s/log4j.rootCategory=INFO, console/log4j.rootCategory=ERROR, console/" \
    $SPARK_HOME/conf/log4j.properties.template > $SPARK_HOME/conf/log4j.properties
```

```python
# 15. wordcounts.py 라는 이름의 새 파일을 만들고 다음 코드를 파일에 추가한다.

# Source code for the 'MapReduce and Word Count' Exercise in
# Data Analytics with Spark Using Python
# by Jeffrey Aven
#
# Execute this program using spark-submit as follows:
#
#  $ spark-submit --master local wordcounts.py \
#     $SPARK_HOME/data/shakespeare.txt \
#     $SPARK_HOME/data/wordcounts
#

import sys, re
from pyspark import SparkConf, SparkContext
conf = SparkConf().setAppName('Word Counts')
sc = SparkContext(conf=conf)

# check command line arguments
if (len(sys.argv) != 3):
   print("""\
This program will count occurances of each word in a document or documents
and return the counts sorted by the most frequently occuring words

Usage:  wordcounts.py <input_file_or_dir> <output_dir>
""")
   sys.exit(0)
else:
   inputpath = sys.argv[1]
   outputdir = sys.argv[2]

# count and sort word occurances 
wordcounts = sc.textFile("file://" + inputpath) \
            .filter(lambda line: len(line) > 0) \
            .flatMap(lambda line: re.split('\W+', line)) \
            .filter(lambda word: len(word) > 0) \
            .map(lambda word:(word.lower(),1)) \
            .reduceByKey(lambda v1, v2: v1 + v2) \
            .map(lambda x: (x[1],x[0])) \
            .sortByKey(ascending=False) \
            .persist()
            
wordcounts.saveAsTextFile("file://" + outputdir)
top5words = wordcounts.take(5)
justwords = []
for wordsandcounts in top5words:
   justwords.append(wordsandcounts[1])
print("The top five words are : " + str(justwords))
print("Check the complete output in " + outputdir)
```

```
# 16. 다음 명령을 사용해 프로그램을 실행한다.

$ spark-submit \
   --master local \
   wordcounts.py \
   $SPARK_HOME/data/shakespeare.txt \
   $SPARK_HOME/data/wordcounts

콘솔에 상위 5개 단어가 표시되어야 한다.
출력 디렉토리는 $ SPARK_HOME/data/wordcounts; 를 확인하자.
이 연습에서는 파티션 하나만 사용했으므로 이 디렉토리에는 하나의 파일(part-00000)이 나타난다.
2개 이상의 파티션을 사용하면 추가 파일(part-00001, part-00002)이 표시된다.

# 17. 16단계의 명령을 다시 실행하자. wordcounts 디렉토리가 이미 존재하기 때문에 덮어 쓸 수 없으므로 실패할 수 있다. 이것은 단순히 디렉토리를 제거하거나 이름을 바꾸면 쉽게 해결된다.
```

...

[[top](#contents)]

<br>

<div id='5.'/>

## 5. 스파크 코어 API를 사용한 고급 프로그래밍

### 5.1. 스파크의 공유변수

#### Broadcast 변수

* 스파크 드라이버 프로그램에 의해 설정된 read-only 변수이다.
* 스파크 클러스터에서 작업자 노드가 사용할 수 있다.
* 스파크 드라이버에 의해 설정된 후에만 읽을 수 있고, 작업자의 실행자에서 실행되는 모든 작업에 사용할 수 있다.
* BitTorrent를 기반으로 하는 peer-to-peer 공유 프로토콜을 사용하면 작업자 간의 효율적인 공유가 가능하다. 이는 단순히 스파크 드라이버에서 실행자 프로세스로 변수를 직접 푸싱하는 것보다 확장성이 뛰어나다.
* 다음 그림은 broadcast 변수가 초기화되고 작업자 간에 전파되며, 작업 내의 노드에 의해 액세스되는 방법을 보여준다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_5_1.png" width="60%" height="60%"></p>

* Broadcast 변수는 SparkContext 밑에 만들어지며 스파크 응용 프로그램의 콘텍스트에서 객체로 접근할 수 있다.
* broadcast() 구문: `sc.broadcast(value)` 
   * Broadcast() 메소드는 특정 스파크 콘텍스트 내에 Broadcast 객체의 인스턴스를 생성한다.
   * value는 Broadcast 객체에서 직렬화되고 캡슐화되며, 유효한 파이썬 객체이다.
   * 만들어진 변수는 응용 프로그램에서 실행되는 모든 작업에서 사용할 수 있다.

```python
# 코드 5.1. broadcast() 함수를 사용해 브로드캐스트 변수 초기화하기

stations = sc.broadcast({'83':'Mezes Park', '84':'Ryland Park'})
stations
# returns <pyspark.broadcast.Broadcast object at 0x...>
```

* 브로드캐스트 변수는 로컬, 네트워크 또는 분산 파일 시스템의 파일 콘텐츠에서 만들 수 있다.

```python
# 코드 5.2. 파일에서 브로드캐스트 변수 만들기

#< stations.csv >
# 83, Mezes Park, 37, ... 
# 84, Ryland Park, 37, ...

stationfile = '/opt/spark/data/stations.csv'
stationdata = dict(map(lambda x: (x[0], x[1]), \
                   map(lambda x: x.split(','), \
                   open(stationsfile))))
stations = sc.broadcast(stationsdata)
stations.value['83'] # broadcast 변수값을 value() 메소드를 통해 반환
# return 'Mezes Park'
```

* value() 구문: `Broadcast.value()`
   * value 값은 해당 키를 사용해 map의 값에 접근할 수 있는 dict(또는 map)이다.
   * value() 함수는 스파크 프로그램의 map() 또는 filter() 연산의 lambda 함수 내에서 사용할 수 있다.
* unpersist() 구문: `Broadcast.unpersist(blocking=False`
   * 브로드캐스트 객체의 unpersist() 메소드는 존재하는 클러스터의 모든 작업자 메모리에서 브로드캐스트 변수를 제거하는 데 사용된다.
   * blocking 인수는 변수가 모든 노드에서 분리될 때까지 이 연산이 차단돼야 하는지 아니면 연산이 비동기 비차단이 될 수 있는지를 지정한다. 메모리를 즉시 릴리즈해야 하는 경우 이 인수를 True로 설정한다.

```python
# 코드 5.3. unpersist() 메소드

stations = sc.broadcast({'83':'Mezes Park', '84':'Ryland Park'})
stations.value['84']
# returns 'Ryland Park'
stations.unpersist()
# 브로드캐스트 변수는 결국 캐시에서 축출된다.
```

* 브로드캐스트 변수와 관련된 스파크 구성 옵션이 몇 가지 있다. (필요시 살펴보자.)
   * spark.broadcast.compress: 브로드캐스트 변수를 작업자에게 전송하기 전에 압축할지 여부를 결정한다. 기본값은 True(권장)
   * spark.broadcast.blockSize: 브로드캐스트 변수의 각 블록 크기를 지정한다. 기본값은 4MB
   * ...
* 브로드캐스트 변수의 장점은 무엇이고, 어떤 경우에 유요하며 필요할까? 종종 2개의 데이터세트를 결합해 결과 데이터세트를 생성하는데 사용함.
* stations(작은 데이터세트)와 status(큰 데이터세트)를 자연 키인 station_id에 조인하는 문제를 생각해보자.

```python
# 코드 5.4. RDD join()을 사용해 조회 데이터 조인하기
status = sc.textFile('file:///opt/spark/data/bike-share/status') \
           .map(lambda x: x.split(',')) \
           .keyBy(lambda x: x[0])
stations = sc.textFile('file:///opt/spark/data/bike-share/stations') \
             .map(lambda x:x.split(',')) \
             .keyBy(lambda x: x[0])
status.join(stations) \
      .map(lambda x: (x[1][0][3], x[1][1][1], x[1][0][1], x[1][0][2])) \
      .count()
# return 907200
```

* 코드 5.4. 는 비싼 shuffle연산을 초래한다. 따라서, 테이블 변수는 stations 드라이버에서 설정하는 것이 좋다.
* 다음 코드에서 stations는 map() 연산을 구현하는 스파크 작업의 런타임 변수로 사용되며, shuffle에 대한 요구사항을 제거한다.

```python
# 코드 5.5. 드라이버 변수를 사용해 조회 데이터 조인하기

stationsfile = '/opt/spark/data/bike-share/stations/stations.csv' # 여기는 .csv 파일이네?
# sdata 라는 드라이버 변수에 station 데이터 할당
sdata = dict(map(lambda x: (x[0], x[1]), \
             map(lambda x: x.split(','), \
             open(stationsfile))))
status = sc.textFile('file:///opt/spark/data/bike-share/status') \
           .map(lambda x: x.split(',')) \
           .keyBy(lambda x: x[0])
status.map(lambda x:(x[1][3], sdata[x[0]], x[1][1], x[1][2])) \
      .count()
# return 907200
```

* 위 방법은 첫 번째 옵션보다 대부분 더 효과적이지만 확장성이 부족하다. 이 경우 변수는 참조 함수 내 클로서(closure)의 일부로 작업자 노드에서의 데이터 전송 및 복제를 비효율적이며 불필요하게 만든다.
* 따라서, 작은 stations 테이블의 브로드캐스트 변수를 초기화해야 한다.
* Peer-to-peer 복제를 사용해 모든 작업자가 변수를 사용할 수 있게 하고, 단일 복사본은 실행 중인 응용 프로그램에 속한 모든 실행자의 모든 작업에서 사용할 수 있게 한다. 그런 다음 두 번째 옵션처럼 map() 연산의 변수를 사용한다. (다음 코드 참조)

```python
# 코드 5.6. 브로드캐스트 변수를 사용해 조회 데이터 조인하기
stationsfile = '/opt/spark/data/bike-share/stations/stations.csv' # 여기는 .csv 파일이네?
# sdata 라는 드라이버 변수에 station 데이터 할당
sdata = dict(map(lambda x: (x[0], x[1]), \
             map(lambda x: x.split(','), \
             open(stationsfile))))

stations = sc.broadcast(sdata) # sdata를 브래드캐스트
status = sc.textFile('file:///opt/spark/data/bike-share/status') \
           .map(lambda x: x.split(',')) \
           .keyBy(lambda x: x[0])
status.map(lambda x:(x[1][3], stations.value[x[0]], x[1][1], x[1][2])) \
      .count()
# return 907200
```

* 이처럼 브로드캐스트 변수의 사용은 런타인 동안 스파크 클러스터의 서로 다른 노드에서 실행되는 프로세스 사이에 데이터를 공유하는 효율적인 방법이다.
* 브로드캐스트 변수의 특징
   * 브로드캐스트 변수를 사용하면 shuffle 연산이 필요없다.
   * 효율적이고 확장 가능한 peer-to-peer 배포 메커니즘을 사용한다.
   * 작업(job)당 한 번씩 복제하는 대신 작업자(executor)당 한 번씩 데이터를 복제한다. 스파크 응용 프로그램에 수천 개의 작업이 있을 수 있으므로 이것은 매우 중요하다.
   * 많은 작업을 여러 번 다시 할 수 있다.
   * 직렬화된 객체로 효율적으로 읽힌다.


#### Accumulator

* 스파크의 또 다른 유형의 공유변수는 accumulator이다.
* accumulator는 broadcase 변수와 달리 업데이트할 수 있다. (증가되는 숫자 값)
* accumulator는 스파크 프로그래밍에서 여러 가지 방법으로 사용할 수 있는 카운터로 생각할 수 있다.
* accumulator를 사용하면 프로그램이 실행되는 동안 여러 값을 집계할 수 있다.
* accumulator는 드라이버에 의해 설정되고, 각각의 Spark Context에서 작업을 담당하는 실행자에 의해 업데이트된다.
* 드라이버는 보통 accumulator의 최종 값을 프로그램의 끝에서 다시 읽을 수 있게 한다.
* acuumulator는 스파크 응용 프로그램에서 성공적으로 완료된 작업마다 한 번만 업데이트된다.
* 작업자 노드는 accumulator의 업데이트를 드라이버(accumulator 값을 읽을 수 있는 유일한 프로세스)로 다시 보낸다.
* accumulator는 정수 또는 부동 소수점 값을 사용할 수 있다.
* 다음 코드와 그림은 accumulator가 어떻게 생성되고, 업데이트되며, 잃히는지를 보여 준다.

```python
# 코드 5.7. Accumulator 생성 및 액세스

acc = sc.accumulator(0)
def addone(x):
   global acc
   acc += 1
   return x + 1
myrdd = sc.parallelize([1,2,3,4,5])
myrdd.map(lambda x: addone(x)).collect()
# returns [2,3,4,5,6]
print('records processed: ' + str(acc.value))
# returns 'records processed: 5' 
```

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_5_2.png" width="60%" height="60%"></p>

* 구문 - accumulator()
   * sc.accumulator(value, accum_param=None)
   * accumulator() 메소드는 특정 스파크 콘텍스트 내에 Accumulator 객체의 인스턴스를 만들고, value 인수에 의해 지정된 값으로 초기화한다.
   * accum_param 인수는 사용자 정의 accumulator를 정한다.
* 구문 - value()
   * Accumulator.value()
   * value() 메소드는 accumulator 값을 가져온다.
   * 이 방법은 드라이버 프로그램에서만 사용할 수 있다.

##### 사용자 정의 Accumulator

* Spark Context로 작성된 표준 accumulator는 int 및 float를 포함한 기본적인 숫자 데이터 유형을 지원한다.
* 사용자 정의 accumulator는 스칼라 숫자 값이 아닌 다른 유형의 변수에 대해 집계 연산을 수행할 수 있다.
* 사용자 정의 accumulator는 AccumulatorParam 도우미 객체를 사용해 생성된다.
* 이때 수행되는 연산은 결합 및 교환이 가능해야 한다 (요구사항). 즉, 수의 순서와 연산의 순서는 무관하다.
* 사용자 정의 accumulator는 벡터를 리스트나 딕셔너리로 accumulate하는 것이 일반적이다.
* 비수학적 컨텍스트에서 비숫자 연산에 동일한 원리가 개념적으로 적용되는데, 문자열 값을 연결하는 사용자 정의 accumulator를 만드는 경우를 예로 들 수 있다.
* 다음은 벡터를 파이썬 딕셔너리로 합치는 데 사용되는 사용자 정의 accumulator의 예제다 
   * addInPlace() 함수: 사용자 정의 accumulator 데이터 유형의 두 객체에 대해 연산하고 새 값을 반환한다.
   * zero() 함수: map 유형에 빈 map을 제공하는 것처럼 각 유형에 0 값을 제공한다.

```python
# 코드 5.8. 사용자 정의 accumulator

from pyspark import AccumulatorParam
class VectorAccumulatorParam(AccumulatorParam):
   def zero(self, value):
      dict1 = {}
      for i in range(0, len(value)):
         dict1[i] = 0
      return dict1
   def addInPlace(self, val1, val2):
      for i in val1.keys():
         val1[i] += val2[i]
      return val1

rdd1 = sc.parallelize([{0: 0.3, 1: 0.5, 2: 0.4}, {0: 0.2, 1: 0.4, 2: 0.2}])
vector_acc = sc.accumulator({0: 0, 1: 0, 2: 0}, VectorAccumulatorParam())
def mapping_fn(x):
   global vector_acc
   vector_acc += x
# 다른 rdd 처리를 수행해야 한다.
rdd1.foreach(mapping_fn)
print(vector_acc.value)
# returns {0: 0.5, 1: 1.2000000000000002, 2: 0.6000000000000001}
```

##### Accumulator 용도

* 일반적으로 accumulator는 처리된 레코드 수를 계산하거나 조작된 레코드 수를 추적하는 등의 용도로 사용된다.
* 또한, 다른 유형의 레코드를 의도적으로 계산할 때도 사용할 수 있다. (ex. 로그 이벤트의 매핑 중에 발견된 여러 응답 코드의 수)
* 경우에 따라 응용 프로그램 내의 프로세싱을 위해 accumulator를 사용하기도 한다.
* Accumulator에서 잘못된 결과를 초래할 수 있는 경우
   * map() 연산 내에서 결과를 계산하기 위해 add-in-place 연산을 사용하는데, 이렇게 add-in-place 연산을 수행하려는 목적으로 accumulator를 호출할 때와 같이 accumulator가 변환에 사용되는 경우 잘못된 결과가 나올 수 있다.
   * 단계 재시도나 추측 실행으로 인해 accumulator 값이 두 번 이상 카운트될 수 있다.
   * 절대적인 정확성이 필요한 경우, foreach() 액션과 같이 스파크 드라이버에 의해 계산된 액션 내에서만 accumulator를 사용해야 한다.
   * 매우 큰 데이터세트의 개념적이거나 지표적인 계수만 고려한다면, 변환된 accumulator를 업데이트해도 된다. (주의 - 사용자 책임임)



#### Broadcast 변수 및 accumulator 사용 연습

* 이 연습에서는 broadcase 변수를 사용해, 불용어를 제거한 다음, accumulator를 사용해 평균 단어 길이를 계산한다.

```
# 1. 사용할 수 있는 모드(로컬, YARN, 클라이언트 또는 독립실행형)를 선택해 pyspark shell을 연다. (여기서는 로컬 모드에서 단일 인스턴스 스파크 배포를 사용)

$ pyspark --master local average_word_length.py
```

```python
# 2. 내장된 urllib2 모듈을 사용해 책의 S3 버킷에서 영어 불용어 리스트를 가져온 다음, split() 함수를 사용해 리스트로 변환한다.

import urllib.request
stopwordsurl = "https://s3.amazonaws.com/sparkusingpython/stopwords/stop-word-list.csv"
req = urllib.request.Request(stopwordsurl)
with urllib.request.urlopen(req) as response:
   stopwordsdata = response.read().decode('utf-8')
stopwordslist = stopwordsdata.split(',')

# 3. stopwordslist 객체에 대한 브로드캐스트 변수를 만든다.
stopwords = sc.broadcast(stopwordslist)

# 4. 모든 단어의 누적 단어 수 및 누적 총 길이에 대한 accumulator를 초기화한다.
word_count = sc.accumulator(0)
total_len = sc.accumulator(0.0) # float인 이유: 결과의 정밀도를 유지하기 위함. 나중에 나눗셈 연산에서 분자로 사용할 수 있음.

# 5. 단어 수와 전체 단어 길이를 누적하는 함수를 만든다.
def add_values(word, word_count, total_len):
   word_count += 1
   total_len += len(word)

# 6. 셰익스피어 텍스트를 로드하고, 문서의 모든 텍스트를 토큰화하고 정규화한다. stopwords 브로드캐스트 변수를 사용해 불용어를 필터링해 RDD를 만든다.
words = sc.textFile('file:///opt/spark/data/shakespeare.txt') \
   .flatMap(lambda line: line.split()) \
   .map(lambda x: x.lower()) \
   .filter(lambda x: x not in stopwords.value)

# 7. foreach 액션을 사용해 결과 RDD를 반복하고, add_values 함수를 호출한다.
words.foreach(lambda x: add_values(x, word_count, total_len))

# 8. Accumulator 공유변수에서 평균 단어 길이를 계산하고 최종 결과를 표시한다.
avgwordlen = total_len.value/word_count.value
print("Total Number of Words: " + str(word_count.value))
print("Average Word Length: " + str(avgwordlen))
```

```python
# 9. 모든 코드를 average_word_length.py 파일에 넣고, spark-submit을 사용해 프로그램을 실행하자.

#
# Source code for the 'Using Broadcast Variables and Accumulators' Exercise in
# Data Analytics with Spark Using Python
# by Jeffrey Aven
#
#  $ spark-submit --master local average_word_length.py
#

from pyspark import SparkConf, SparkContext
conf = SparkConf().setAppName('Using Broadcast Variables and Accumulators')
sc = SparkContext(conf=conf)

# step 2
import urllib.request
stopwordsurl = "https://s3.amazonaws.com/sparkusingpython/stopwords/stop-word-list.csv"
req = urllib.request.Request(stopwordsurl)
with urllib.request.urlopen(req) as response:
   stopwordsdata = response.read().decode("utf-8") 
stopwordslist = stopwordsdata.split(",")
# step 3
stopwords = sc.broadcast(stopwordslist)
# step 4
word_count = sc.accumulator(0)
total_len = sc.accumulator(0.0)
# step 5
def add_values(word,word_count,total_len):
   word_count += 1
   total_len += len(word)
# step 6
words = sc.textFile('file:///opt/spark/data/shakespeare.txt') \
   .flatMap(lambda line: line.split()) \
   .map(lambda x: x.lower()) \
   .filter(lambda x: x not in stopwords.value)
# step 7
words.foreach(lambda x: add_values(x, word_count, total_len)) 
# step 8
avgwordlen = total_len.value/word_count.value
print("Total Number of Words: " + str(word_count.value))
print("Average Word Length: " + str(avgwordlen))
```
```
$ spark-submit --master local average_word_length.py
```

### 5.2. 스파크의 데이터 파티셔닝

* 파티셔닝은 대부분 스파크 프로세스에 필수적이다.
* 효율적인 파티셔닝은 응용 프로그램 성능을 수십 배 향상시킬 수 있지만, 비효율적인 파티셔닝은 프로그램을 완료하기 힘들게 만든다.
* 과도하게 큰 파티션으로 인해 Executor-out-of-memory 오류와 같은 문제가 발생할 수 있다.
* RDD 파티션 내용을 요약하고, 파티셔닝에 영향을 주거나 파티션 내의 데이터에 보다 효과적으로 액세스할 수 있는 API 메소드르 알아보자.

#### 5.2.1. 파티셔닝 개요

* RDD 변환에서 생성할 파티션 수는 일반적으로 구성할 수 있다.
* 파티셔닝의 몇 가지 기본 동작을 알 필요가 있다.

```python
# HDFS를 사용하면 블록마다 RDD 파티션을 만든다. (일반적으로 HDFS의 블록 크기는 128MB)

myrdd = sc.textFile("hdfs:///dir/filescontaining10blocks")
myrdd.getNumPartitions()
# returns 10
```
```python
# ByKey 연산(groupByKey(), reduceByKey()) 같은 셔플연산과 
# numPartitions 값이 메소드에 대한 인수로 제공되지 않는 다른 연산들은 
# spark.default.parallelism 구성 값과 같은 파티션 개수를 만든다.

# with spark.default.parallelism=4
myrdd = sc.textFile("hdfs:///dir/filescontaining10blocks")
mynewrdd = myrdd.flatMap(lambda x: x.split()) \
   .map(lambda x: (x,1)) \
   .reduceByKey(lambda x, y: x+y)
mynewrdd.getNumPartitions()
# returns 4
```
```python
# spark.default.parallelism 구성 매개변수가 설정되지 않은 경우,
# 변환에 의해 만들어지는 파티션 수는 현재 RDD 리니지의 업스트림 RDD에 의해 정의된 파티션의 최대 수와 같다.

# spark.default.parallelism이 설정되지 않은 상태
myrdd = sc.textFile("hdfs:///dir/filescontaining10blocks")
mynewrdd = myrdd.flatMap(lambda x: x.split()) \
   .map(lambda x: (x,1)) \
   .reduceByKey(lambda x, y: x+y)
mynewrdd.getNumPartitions()
# returns 10
```

* 스파크에서 사용하는 기본 파티션 클래스는 HashPartitioner이다. 
   * 이는 결정적 해시 함수를 사용해 모든 키를 해시한 다음, 키 해시를 사용해 거의 동일한 버킷을 만든다.
   * 목표는 키를 기반으로 지정된 파티션 수에 데이터를 균등하게 분산시키는 것이다.
* filter() 변환과 같은 일부 스파크 변환은 결과 RDD의 분할 동작을 변경할 수 없다.
   * 예를 들어, 4개의 파티션이 있는 RDD에 filter() 함수를 적용하면, 원래의 RDD와 동일한 분할 스키마(해시 분할)를 사용해, 4개의 파티션이 있는 필터링된 새 RDD를 생성한다.
* 기본 동작은 일반적으로 사용할 수 있지만, 어떤 경우에는 비효율적이다.
   * 다행히도 스파크는 이러한 잠재적 문제를 해결할 수 있는 몇 가지 메커니즘을 제공한다.

#### 5.2.2. 파티션 제어

* RDD에는 몇 개의 파티션이 있어야 할까?
* 문제는 다음 두 가지 양극단의 스펙트럼에서 발생한다.
   * 매우 적은 파티션 개수를 가진다면, 매우 큰 크기의 파티션이 실행자에게 메모리 부족 문제가 발생한다.
   * 매우 많은 파티션 개수를 가진다면, 매우 작은 크기의 파티션 때문에 작은 입력에도 작업이 발생한다. (쓸데없이 작업량이 증가됨)
* 크고 작은 파티션을 혼합하면, 추측 실행(speculative execution)이 불필요하게 발생한다.
   * 추측 실행은 클러스터 스케줄러가 늘리게 실행되는 프로세스를 선점하기 위해 사용되는 메커니즘이다.
* 스파크 응용 프로그램에서 하나 이상의 프로세스가 느려지는 근본적인 원인이 비효율적인 파티셔닝이라면, 추측 실행은 도움이 되지 않는다.

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_5_3.png" width="60%" height="60%"></p>

* filter() 연산은 필터 조건을 만족하는 레코드만 사용해 일대일 기반으로 모든 입력 파티션에 대해 새 파티션을 만든다. (위 그림)
* 이로 인해 일부 파티션의 데이터가 다른 파티션보다 훨씬 적어 데이터 왜곡, 추측 실행 가능성 및 다음 단계에서 최적이 아닌 성능과 같은 나쁜 결과를 초래할 수 있다.
* 이러한 경우 스파크 API에서 재분할 방법 중 하나를 사용할 수 있다.
   * partitionBy(), coalesce(), repartition(), repartitionAndSortWithinPartitions()
   * 이 함수들은 파티션이 완료된 입력 RDD를 가지고 n개의 파티션을 가진 새로운 RDD를 생성한다.
   * 여기서 n은 원래 파티션 수보다 많거나 적을 수 있다.
* repartition() 함수는 4개의 고르지 않은 분산 파티션을 기본 'HashPartitioner'를 사용해 2개의 '고르게' 분산된 파티션으로 통합하는 데 적용된다. (아래 그림)

<p align="center"><img src="https://github.com/gritmind/my-review-notes/blob/master/code/book/spark_using_python/images/pic_5_4.png" width="60%" height="60%"></p>

* 최적의 파티션 수 결정
   * 최적의 파티션 수를 결정할 때 종종 반환값을 줄이는 시점(각 추가 파티션이 성능을 떨어뜨리기 시작하는 시점)을 찾을 때까지 다른 값들을 실험해야 한다.
   * 시작점에서의 간단한 공리는 클러스터 코어 수의 2배, 즉, 모든 작업자 노드의 총 코어 수의 2배를 사용하는 것이다.
   * 데이터세트가 변경되면 사용되는 파티션 수를 다시 검토하는 것이 좋다.

#### 5.1.3. Repartitioning Functions

* RDD를 재분할하는 데 사용되는 주요 함수는 다음과 같다.
<br>

* `partitionBy()`
   * 구문: `RDD.partitionBy(numPartitions, partitionFunc=protable_hash)`
   * paritionBy() 메소드는 기본적으로 portable_hash 함수(HashPartitioner)를 사용해 입력 RDD와 동일한 데이터를 포함하지만, numParitions 인수로 지정된 파티션 수와 함께 새 RDD를 반환한다. 

```python
# partitionBy 함수

kvrdd = sc.parallelize([(1, 'A'), (2, 'B'), (3, 'C'), (4, 'D')], 4)
kvrdd = getNumPartitions()
# return 4
kvrdd.partitionBy(2).getNumPartitions()
# return 2
```

* partitionBy() 함수는 protable_hash 함수 대신 rangePartitioner를 사용해 paritionBy()를 호출하는 다른 함수(ex. sortByKey())에 의해 호출된다.
* rangePartitioner는 키별로 정렬된 레코드를 동일한 크기의 범위로 분할한다. 이것은 해시 파티셔닝의 대안이다.
* partitionBy() 변환은 웹 로그를 매월 파티션으로 bucket하는 함수와 같이 사용자 정의 파티셔너를 구현하는 데 유용한 함수이기도 하다.
* 사용자 정의 파티션 함수는 키를 입력으로 가져와 partitionBy() 함수에 지정된 numPartitions와 0사이의 숫자를 반환한 다음 해당 반환값을 사용해 요소를 대상 파티션으로 전달한다.
<br>

* `repartition()`
   * 구문: `RDD.repartition(numPartitions)`
   * repartition() 메소드는 입력 RDD와 동일한 데이터를 갖는 새로운 RDD를 반환하며 numPartitions 인수로 지정된 파티션 수와 정확하게 일치한다.
   * repartition() 메소드는 셔플을 필요로 할 수 있으며, partitionBy()와 달리 파티셔너 또는 분할 함수를 변경하는 옵션이 없다.
   * repartition() 메소드를 사용하면 입력 RDD에 있는 것보다 대상 RDD에 더 많은 파티션을 만들 수 있다.

```python
# 코드 5.10. repartition() 함수

kvrdd = sc.parallelize([(1, 'A'), (2, 'B'), (3, 'C'), (4, 'D')], 4)
kvrdd.repartition(2).getNumPartitions()
# return 2
```

* `coalesce()`
   * 구문: `RDD.coalesce(numPartitions, shuffle=False)`
   * ...

...


### 5.3. RDD 저장 옵션

* 지금까지 RDD를 클러스터 작업자 노드의 메모리에 있는 분산 불변 객체 모음으로 설명했다.
* 그러나 여러 가지로 유익한 RDD의 다른 저장 옵션이 있다.
* 다양한 스토리지 레벨과 캐싱 및 지속성을 보기 전에 RDD 리니지 개념을 먼저 보자.

#### 5.3.1. RDD Lineage 재검토






[[top](#contents)]

<br>

<div id='6.'/>

## 6. 스파크로 SQL 및 NoSQL 프로그래밍하기

...


[[top](#contents)]

<br>

<div id='7.'/>

## 7. 스파크를 사용한 스트림 처리 및 메시징

...

[[top](#contents)]

<br>

<div id='8.'/>

## 8. 스파크를 사용한 데이터 과학 및 머신 러닝 소개

...


