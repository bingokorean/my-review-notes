# 알기 쉽게 해설한 JAVA

김충석

### Contents
1.	[컴퓨터와 프로그램 그리고 자바](#컴퓨터와-프로그램-그리고-자바)
2.	[자바의 환경](#자바의-환경)
3.	[자바의 기본 구조1: 변수, 자료형, 연산자](#자바의-기본-구조1-변수-자료형-연산자)
4.	[자바의 기본 구조2: 선택, 반복, 배열](#자바의-기본-구조2-선택-반복-배열)
5.	[객체지향 개념](#객체지향-개념)
6.	[클래스: 속성](#클래스-속성)
7.	[클래스: 기능](#클래스-기능)
8.	[상속](#상속)
9.	[인터페이스와 예외처리](#인터페이스와-예외처리)
10.	~~다중 스레드~~
11.	~~패키지와 주요 클래스~~
12.	[입출력](#입출력)
13.	~~네트워킹~~
14.	~~그래피컬 사용자 인터페이스~~
15.	~~이벤트 처리~~
16.	~~스윙~~


## 컴퓨터와 프로그램 그리고 자바

개발자가 작성한 프로그램이 컴퓨터에서 실행되기 위해서 무조건 컴퓨터가 인식할 수 있는 0과 1의 형태로 번역되어야 한다. 프로그램을 번역하여 실행하는 방법은 다음과 같다.

1. 컴파일(Compile) 기법
   * 0과 1로 번역해주는 컴파일러가 반드시 필요함. (ex. C언어로 작성된 프로그램을 실행시키기 위해서는 C컴파일러가 반드시 필요)
   * 컴파일러에 의해 번역된 프로그램은 언제든지 실행될 수 있는 프로그램임. 
   * (장점) 컴파일되어 실행파일만 되기만 하면 다음부터는 번역 절차 없이 실행 파일만 실행시키므로 실행시간의 효율성이 뛰어남.
   * (단점) 특정 시스템(ex. 윈도우)에서 번역된 실행파일은 다른 시스템(ex. 리눅스)에서는 실행되지 않음. 다른 시스템에서 실행하기 위해서는 그 시스템에서 다시 번역작업을 수행하여 실행파일을 생성해야 함.
   * 대표적인 컴파일 기법의 언어로 C언어가 있음.

2. 인터프리트(Interprete) 기법
   * 인터프리터에 프로그램을 실행시키는 방법. 컴파일과 다르게 0과 1로 구성된 실행파일을 생성하지 않음.
   * 프로그램을 직접 한 줄씩 번역한 다음 바로 실행시켜 결과를 나타냄.
   * 주로 스크립트 언어(ex. HTML, Javascript, ASP, PHP, Perl, Python)들이 대부분 인터프리트 기법을 사용.
   * (장점) 컴파일러 방법에 비해 배우기 쉽고, 이식성이 뛰어남.
   * (단점) 프로그램 자체가 공개되고 실행시간이 느림.

3. 하이브리드(Hybrid) 기법
   * 컴파일 기법과 인터프리트 기법을 모두 사용함.
   * 특징은 중간코드(ex. 바이트 코드 (=.class 파일))를 사용함. 중간코드는 다양한 형태의 서로 다른 시스템에서 인터프리터에 의해 직접 실행됨.
   * 컴퓨터, 운영체제에 상관없이 실행되는 하드웨어에 독립적인 코드
   * (장점1) 인터프리터 단점인 소스코드 공개하지 않아도 됨. 컴파일러 단점인 특정 컴퓨터에 종속성을 벗어남.
   * (장점2) 한 번 작성된 프로그램은 어떤 컴퓨터 시스템에서든지 즉시 실행될 수 있어 이식성이 매우 높음. 이러한 이식성은 네트워크(인터넷) 환경에서 특히 강한 면모를 보임.
   * 자바, C# 등이 있음.
 
자바는 완전한 객체지향 언어(Object Oriented Language)이며, 객체지향의 특성인 클래스, 상속, 캡슐화, 다형성 등의 개념이 잘 적용된 언어이다. 객체지향 프로그래밍은 우리가 살아가는 실세계와 동일한 사고방식의 프로그램이다. 자바의 객체지향 모델은 단순하며 쉽게 확장될 수 있는 개념을 가진다. 객체지향에서는 모든 것을 객체로 표현하지만, 자바 언어는 정수 또는 실수와 같이 많이 사용되는 요소들을 객체는 물론 성능을 높이기 위해 기본 자료형(primitive type)으로도 제공하고 있다. 

자바는 운영체제 독립적이다. 자바 프로그램은 JVM(Java Virtual Machine)이 구축된 컴퓨터에서는 어디에서든지 실행시킬 수 있다. JVM은 소프트웨어로 구성된 에뮬레이터(Emulator: 대리실행기)로서 운영체제와 자바 프로그램 사이에서 자바 프로그램이 실행될 수 있는 환경을 제공한다. 

## 자바의 환경

자바 언어를 사용하여 작성할 수 있는 프로그램의 형태는 매우 다양하며, 광범위한 적용 범위를 가지고 있다. 또한, 자바 개발 환경도 작성하는 프로그램의 특성(적용 분야와 규모)에 따라 다르게 제공되고 있다. 일반적으로 자바 프로그램은 다음과 같은 형태로 구분될 수 있다.

* 자바 응용 프로그램: C 프로그램과 같이 일반적인 응용 프로그램이다. 사용자는 CUI나 GUI와 같은 인터페이스를 이용하여 프로그램을 사용하게 된다. 
* 자바 애플릿: 웹 검색기상에서 작동하는 프로그램이다. 웹의 표준 언어인 HTML로 애플릿을 지정하면, 웹서버로부터 애플릿이 네트워크를 통하여 검색기로 다운로드 되어 실행되는 형태이다. 웹 검색기에는 애플릿을 해석하여 실행할 수 있는 인터프리터가 내장되어 있다.
* 자바 서블릿(Servlet): 웹 환경에서 실행되는 자바 프로그램이다. 서버에서 실행되는 프로그램으로 웹 검색기에서 실행되는 응용에 적합하다. 서버의 실행 결과를 웹 클라이언트가 볼 수 있는 HTML로 만들어 나타내는 프로그램이다. 주로 데이터베이스와 연동하는 프로그램이 서블릿으로 작성된다.
* JSP(Java Server Page): 서블릿과 비슷한 형태이나, HTML 속에 자바 코드를 삽입하여 사용하는 형태이다.
* 자바 빈스(Beans): 자바로 작성한 프로그램들을 부품처럼 사용하여 프로그래밍하는 방법이다. 주로 대규모의 프로그램 개발 시 사용하는 방법이다.


<자바노트)> <br>

* Stack영역(지역변수), Heap영역(동적으로 할당되는 부분 - C언어: malloc( )-free( ), C++언어: new-delete), Data영역(전역변수, 정적변수)
* Java에서는 실행시키기 위해서는 반드시 객체를 생성해야 하는데, 객체를 생성하지 않고 바로 메모리상에 할당하기 위해서 static 키워드를 사용한다. (static만 붙어있으면 객체를 생성하지 않아도 된다) 
* 하나의 문자(character)를 표현할 때, C 언어는 7비트 아스키코드를 사용하지만, JAVA는 16비트 유니코드를 사용한다. 하나의 파일 안에 public이 2개 들어갈 수는 없다. 일반적으로 main 함수가 있는 쪽에 public이 들어간다. 하나의 파일에 여러 개의 class가 있는 경우. 여러 개의 class 중에서 하나의 class에만 public을 붙일 수 있다.Public이 붙은 class가 있으면 그 class 이름으로 파일명을 작성해서 저장해야 한다. (그러나 실행 시에는 main 함수가 있는 class 이름으로 실행해야 한다)


## 자바의 기본 구조1: 변수, 자료형, 연산자

변수는 값을 가진다. 변수가 가질 수 있는 값의 형태를 자료형(Data Type)이라고 한다. 자바의 자료형은 크게 2가지로 구분될 수 있다. 
* 기본 자료형(primitive type) – int a = 10; 기본 자료형으로서 변수로 지정된 위치에 **값**이 저장됨
* 참조 자료형(reference type) – Integer b = new Integer(10); 참조 자료형으로서 변수로 지정된 위치에는 실제 값이 있는 곳의 **주소**가 저장됨.

자바는 객체지향 언어이기 때문에 프로그램 내의 모든 요소들을 객체로 표현할 수 있지만, 실행시간의 효율성을 위해 가장 많이 사용되는 8개의 자료형을 기본 자료형으로 제공한다. (기본 자료형: 정수형(byte, short, int, long), 실수형(float, double), 논리형(Boolean), 문자형(char))

자바의 4가지 정수형은 최상위 비트를 부호비트로 사용하고 있다. 음수표기법은 다음과 같이 2가지 방식이 있다. (둘 다 최상위 비트를 부호비트로 사용하는 것은 동일하다)
* (1) 부호비트에 의한 음수표기법
* (2) 2의 보수를 사용한 음수표기법 - 모든 비트를 반전하고 +1을 한다

**2의 보수법을 사용한 음수 표기법 예제** <br>
만약, byte 자료형에서 예를 들어, 1000 0000을 보면, 최상위 비트가 1이므로 음수로 취급한다. 여기서 십진수로 바꾸기 위해서 2의 보수법을 사용한다. 1000 0000의 반전인 0111 1111에서 +1을 해서 1000 0000이 나온다. 십진수로 해석하면 128이고, 음수이기 때문에 -128이라는 최종 결과가 나온다. (-128인데 인코딩이 1000 0000으로 된다는 것을 기억하자; 이렇게 최상위 비트가 부호비트 이므로 8비트 정수 범위가 -128~127이 되는 것이다)

자바에서는 하나의 문자를 나타낼 수 있는 char형을 기본 자료형으로 제공한다. 기존의 언어들이 대부분 사용하고 있는 아스키코드(8비트)가 아닌 유니코드(16비트)를 사용한다. 유니코드를 사용함으로써 자바는 세계 다양한 나라들의 모든 언어를 나타낼 수 있다. (최대 65,536개의 문자)

자바에서는 특수한 문자를 나타내기 위해 역슬래시(backslash)를 사용한다.

```java
public class PrimitiveType {
	public static void main(String args[ ])
	{
		byte a = 127;
		short b = 32767;
		int c = 2137483647;
		long d = 9223372036854775807L;
		float a = 0.12345678901234567890f;
		double b = 0.12345678901234567890;
		char grade = ‘A’;
		char grade1 = ‘\u0041’;		// ‘A’와 같은 의미의 유니코드
		char year = ‘2018’;
		char name = ‘김영수’;
		char char1 = ‘\t’;
		char char2 = ‘\n’;
		System.out.println(…);
	}
}
```

비트(bit) 연산자 <br>
비트 연산자는 비트 단위로 연산할 수 있는 연산자이다. 비트 단위의 연산은 정수형의 데이터에만 적용이 가능하다. 

문자열(String) <br>
자바에서의 문자열은 기본 자료형으로 제공되고 있지 않고, String 클래스로 구현되어 있다. 빈번하게 사용되기 때문에 기본 자료형과 같이 변수로 사용될 수 있다.

```java
public class StringTest {
	public static void main(String args[ ]) {
		String str1 = “대한민국”;
		String str2 = new String(“대한민국!”); 	// 원래 이렇게 new로 선언. 
  }
}
```
여기서 str1과 str2는 참조 자료형으로 주소가 들어간다 (C언어와 비교하면 \*이 있는 것과 같다)

## 자바의 기본 구조2: 선택, 반복, 배열

자바에서의 배열을 사용하기 위해서는 배열을 선언하고, 생성해주어야 한다.
```java
type name[ ] = new type[size]
type name[ ][ ] = new type[size][size];
type name[ ][ ][ ] = new type[size][size][size];
```

배열의 첫 번째 axis는 필수적으로 size를 정해줘야 한다. 나머지 axis들은 size를 정해줘도 되고 안정해줘도 된다.

```java
int[ ][ ] score = new int[2][ ];
score[0] = new int[2];
score[1] = new int[3];
score[2] = new int[4];
```

위와 같이 2차원 배열이지만 각 행의 요소의 개수가 다르게 생성할 수 있다. (C언어에서는 2차원 배열의 열의 개수가 일정하게 생성해야 하지만, Java에서는 유동적으로 원하는 개수만큼 생성할 수 있다)

배열의 선언과 생성과정을 거치면 배열을 사용하기 위해 초기화를 해야한다.

```java
int id[ ] = new int[3];
int[0] = 20119501;
int[1] = 20119502;
int[2] = 20119503;

int id[ ] = { 20119501, 20119502, 20119503 };	// 이렇게 한 문장으로 하는 게 좋다
```

위와 같이 한 문장으로 생성과 초기화 과정을 다 거치는 형태가 C언어랑 동일하다.

그런데… C언어는 `int id[3];` 이 되는데, 자바에서는 `int id[3];` 은 에러가 뜬다 (`int id[ ];` 는 또 괜찮다). 왜 그럴까? C언어는 변수를 선언하기만 해도 메모리에 할당이 되지만, 자바는 객체지향 언어로서 (기본 자료형은 제외하고) new로 생성해야 메모리에 할당이 된다. (자바에서 배열을 사용하기 위해서 반드시 선언과정과 생성과정이 필요하다)

## 객체지향 개념

자바는 객체지향(Object-Oriented) 언어이다. 객체지향 이론은 컴퓨터를 통하여 실세계와 같은 환경을 흉내(simulation)내기 위해 발전한 이론이다. 실세계는 사물(객체)로 구성되어 있으며, 이러한 사물들이 상호작용에 의해 실세계는 작동한다. 실세계의 사물을 분석해보면 사물은 “속성+기능”으로 구성되어 있음을 알 수 있다. 객체지향 이론은 실세계의 모든 사물들을 속성과 기능으로 정의하고, 사물들 간의 상호작용을 정의하여 실제 세계를 흉내내는 이론이다. 객체지향 이론은 1960년대 클래스(class), 상속(inheritance), 상속화(encapsulation), 다형성(polymorphism) 등의 개념을 중심으로 발전하였다.객체지향 언어의 개념과 대비되는 언어로서 절차지향(procedural-oriented) 언어가 있다.

절차지향 언어로 작성된 프로그램에서 프로그램의 기본 단위는 절차(procedure) 또는 함수(function)로 정의된다. 이러한 절차나 함수는 기능을 정의하는데 사용되며, 공통으로 사용되는 속성(데이터)들은 절차나 함수에 의해 공유되는 형태이다. 이러한 형태에서는 공유되는 속성(데이터)의 형태가 바뀌면 모든 함수의 내용이 변경되어야 하는 결정적 단점을 가지고 있다. 반면, 객체지향 언어에서는 프로그램의 기본단위가 객체이며, 객체는 “속성+기능”으로 구성된다. 객체지향 언어에서는 모든 요소들을 반드시 객체로 표현해야 하며, 프로그래밍은 이러한 객체의 생성과 생성된 객체의 상호관계를 설정하는 것으로 이루어진다.

**객체지향의 장점** <br>
객체지향 프로그램의 개념은 우리들의 실생활과 같은 개념의 프로그램 방식을 제공한다.
  * 문제를 쉽고 자연스럽게 프로그램화(모델링) 할 수 있다 (클래스, 캡슐화, 다형성 등을 통해)
  * 쉬운 프로그램의 개발로 인한 생산성을 향상시킬 수 있다. (객체들은 독립성을 가짐. 객체들을 서로 연결하여 프로그램을 완성할 수 있음)
  * 프로그램 모듈을 재사용할 수 있다. (독립된 모듈은 다양한 프로그램에서 재사용될 수 있음)

**클래스** <br>
객체지향에서는 동일한 속성과 기능을 가진 객체를 생성하기 위해 클래스라는 형판(template)을 제공하고 있다. 즉, 클래스는 하나의 클래스로부터 여러 개의 객체를 생성하기 위해 사용하는 형판 즉, 틀이라 보면 된다. 객체는 “속성+기능”으로 구성되기 때문에 객체를 생성하는 클래스 역시 “속성+기능”으로 구성된다. (속성: 변수, 기능: 함수)

**객체** <br>
클래스로부터 객체를 생성하는 과정을 실체화(instantiation)라고 하고, 객체를 인스턴스(instance)라 부르기도 한다. 하나의 클래스로부터 객체가 생성될 때 각 객체는 같은 속성과 기능을 가지지만, 속성에 저장된 값은 모두 다르게 지정할 수 있다.

```java
class Avg {
	String name;
	int avg;
	public String average(int kor, int eng) {
		avg = (kor + eng) / 2;
		return name + avg
	}
}

public class AvgTest {
	public static void main(String[ ] args) {
		Avg student1 = new Avg( );
		Avg student2 = new Avg( );
		student1.name = “김철수”;
		student2.name = “김영희”;
		String str1_avg = student1.average(70, 80);
		String str2_avg = student2.average(80, 90);
		System.out.println(str1_avg);
		System.out.println(str2_avg);
	}
}
```

**상속(Inheritance)** <br>
student 객체와는 비슷하지만 속성과 함수가 약간 다른 객체를 생성하려면 어떻게 해야하나? 처음부터 다 설계해야 할까? 아니다! 객체지향이니까, 기존에 있는 student 객체를 활용할 수 있다. 어떻게? 상속을 이용한다. 즉, 기존 클래스로부터 모든 속성과 메소드를 상속받고, 더 필요한 속성과 메소드를 추가하여 새로운 클래스를 생성할 수 있다. 이러한 개념이 상속이다.

```java
class Avg {
	String name;
	int avg;
	public String average(int kor, int eng) {
		avg = (kor + eng) / 2;
		return name + avg
	}
}
class AvgTotal extends Avg {			// extends 는 import 와 느낌이 비슷하다
	public int total(int kor, int eng) {
		int score = kor + eng;
		return score;
	}
}
public class AvgTest2 {
	public static void main(String[ ] args) {
		AvgTotal student1 = new AvgTotal( );
		AvgTotal student2 = new AvgTotal( );
		student1.name = “김철수”;			// 상속한 Avg 내에 있는 name에 접근한다
		student2.name = “김영희”;
		String str1_avg = student1.average(70, 80);
		String str2_avg = student2.average(80, 90);
		int st1_total = student1.total(70, 80);
		int st2_total = student2.total(80, 90);
		System.out.println(str1_avg+” 총점=”+st1_total);
		System.out.println(str2_avg+” 총점=”+st2_total);
	}
}
```

클래스의 상속은 확장(entend)의 개념을 가진다. 즉, 상위 클래스의 모든 것을 상속받고 추가로 더 가지는 클래스를 구성하는 것이 상속이다. 상위 클래스로 갈수록 general 해지고, 하위 클래스로 갈수록 specific 해진다.

클래스들 사이의 상속은 소프트웨어 설계를 간단하게 할 수 있는 이점을 제공한다. 즉, 기존의 클래스로부터 모든 요소를 상속받고 새로운 클래스에는 추가되는 자료구조와 메소드만 지정하면 된다. 상속의 개념은 코드를 간결하게 하고, 코드의 재사용성(resusing)을 높인다.

다수의 클래스로부터 상속받아 새로운 클래스를 생성하는 경우도 있다. 이를 다중상속(multiple inheritance)라 한다. 자바는 상속관계에서 하나의 상위 클래스만 허용하며, 다중상속은 허용하지 않는다. (인터페이스를 통해 다중상속을 흉내낼 수 있다)

**캡슐화(Encapsulation)** <br>
객체는 속성과 속성을 처리하는 메소드를 가지고 있다. 객체를 사용하는 쪽에서는 그 객체의 인터페이스만 알면 그 객체를 충분히 사용할 수 있다. 객체가 실제 데이터를 어떻게 처리하는지는 알 필요가 없고, 실제 처리방법은 숨겨져야 한다. 이러한 개념이 캡슐화이다. (캡슐화와 다중상속을 흉내내기 위해서 인터페이스가 존재?) (캡슐화를 잘 이해하면 전체적인 프로세스를 이해할 때 개념단위로 이해할 수 있다?)

클래스를 작성할 때 프로그램 작성자는 숨겨야 하는 정보(private)와 공개해야 하는 정보(public)를 구분하여 기술할 수 있다. 객체를 사용하는 사람은 객체 중에 공개하는 정보에만 접근할 수 있다. 이러한 기법을 제공함으로써 객체의 사용자로부터 정보를 은폐(information hiding)할 수 있다.

캡슐화를 통한 정보의 은폐
* 객체에 포함된 정보의 손상과 오용을 막을 수 있다
* 객체 내부의 조작 방법이 바뀌어도 사용방법은 바뀌지 않는다
* 데이터가 바뀌어도 다른 객체에 영향을 주지 않아 독립성이 유지된다
* 처리된 결과만 사용하므로 객체의 이식성이 좋다
* 객체를 부품화 할 수 있어 새로운 시스템의 구성에 부품처럼 사용할 수 있다

**메시지(Message)** <br>
메시지는 객체에 일을 시키는 행위라 할 수 있다. 프로그램에서 생성된 객체들은 이러한 메시지를 주고받음으로써 일을 수행한다. 프로그램 작성자는 사용하고자 하는 객체를 정의한 다음 이러한 객체들이 어떤 일을 수행해야 하는지를 메시지로 기술해야 한다. 일반적으로 메시지에는 메시지를 받을 객체의 이름, 메소드 이름, 메소드의 수행에 필요한 인자(argument)들을 포함한다. (메시지는 그냥 객체에 있는 함수 호출과 같다)

```java
public class AvgTest {
	public static void main (String[] args) {
		AvgTotal student1 = new AvgTotal( );
		String str1_avg = student1.average(70, 80);	// 객체의 메소드 호출 메시지
		int st1_total = student1.total(70, 80);	        // 객체의 메소드 호출 메시지
	}
}
```

**다형성(Polymorphism)** <br>
다양한(poly) 변신(morphism)을 의미하는 그리스어에 기원을 둔다. 즉, 서로 다른 객체가 동일한 메시지에 대하여 서로 다른 방법으로 응답할 수 있는 기능이다. “서로 다른 객체”, “동일한 메시지”, “서로 다른 방법”

```java
class Avg3 {
	public String name;
	private int avg;
	public String average(int kor, int eng) {	// 매개변수가 2개일 때 수행
		avg = (kor + eng) / 2;
		return name + “ 두 과목 평균 : “ + avg;
	}
	public String average(int kor, int eng, int mat) {	// 매개변수가 3개일 때 수행.
		avg = (kor+eng+mat) / 3;
		return name+ “ 세 과목 평균 : “+ avg;
	}
}
public class AvgTest3 {
	public static void main(String[ ] args) {
		Avg3 student1 = new Avg3( );
		Avg3 student2 = new Avg3( );
		student1.name = “김철수”;
		student2.name = “김영희”;
		String st1_avg = student1.average(70, 80);           // 다른 객체 동일 메시지
		String st2_avg = student2.average(70, 80, 90);	     // 다른 객체 동일 메시지
		System.out.println(st1_avg);
		System.out.println(st2_avg);
	}
}
```

## 클래스: 속성
자바 프로그램은 클래스로부터 객체를 생성하여 프로그램이 작성된다. 객체를 생성하기 위해서 클래스를 먼저 작성하여야 한다. (자바 프로그램은 클래스의 집합이다) 클래스는 클래스의 속성에 해당하는 멤버 변수 부분과 클래스의 기능에 해당하는 생성자 (생성자 메소드라고도 부름)와 메소드 정의부분으로 구성된다.

```java
public class Box_Sample {
	int width;
	int height;
	int depth;
	int volume;

	Box_Sample(int w, int h, int d) {	// 생성자: 클래스 이름과 같고 반환형 타입이 없다
		width = w;
		height = h;
		depth = d;
	}

	int volume_compute( ) {		        // 메소드
		volume = width * height * depth;
		return volume;
	}
}
```

클래스를 선언할 때, 클래스의 특성을 나타내는 한정자를 지정하여 선언할 수 있다.

```
예) [public/final/abstract] class Class-name {
	…클래스의 속성과 기능을 기술
}
```

클래스는 다음과 같은 한정자를 가진다
* public: public 한정자는 모든 클래스에서 접근 가능 (클래스로부터 객체 생성 가능) 함을 의미한다.
* 한정자 사용안함: 한정자를 사용하지 않고 선언된 클래스는 같은 패키지 내의 클래스에서만 접근 가능함을 의미한다.
* final: final은 서브 클래스를 가질 수 없는 클래스를 말한다. 즉, final로 선언된 클래스로부터는 새로운 클래스가 상속되어 생성될 수 없음을 의미한다. 현재의 클래스를 다른 클래스에서 상속받지 못하도록 하는 것은 정보 보호 측면에서 유용하다.
* abstract: 추상(abstract) 클래스를 의미한다. 추상 클래스는 객체를 생성할 수 없는 클래스이다.

자바에서 하나의 프로그램에는 하나의 클래스만을 정의하는 것이 원칙이다. 만일 여러 개의 클래스가 하나의 프로그램에 정의된다면 public 한정자는 한 클래스에만 사용해야 한다. 자바 응용 프로그램인 경우에는 main() 메소드가 있는 클래스에 public을 사용해야 한다 (모든 클래스에 한정자를 지정하지 않으면 main() 메소드가 있는 클래스를 public으로 취급한다). 만일 main() 메소드가 있는 클래스가 아닌 다른 클래스에 public 한정자를 사용한다면 자바 컴파일러가 오류를 발생시킨다. 

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/java_kim/images/1.png" width="70%" height="70%"></p>

**객체 선언과 생성** <br>
객체를 생성하기 위해서는 우선 객체를 선언해야 한다. 자바에서는 사용될 모든 변수나 객체들에 대해 미리 선언하는 것을 요구한다.

```java
Box mybox1;
Avg student1;
String name;
```

객체의 선언만으로 객체가 생성되지 않는다. 객체가 메모리상에서 생성되기 위해서는 선언된 객체를 명시적으로 생성시켜야 한다.

```java
Box mybox1;
mybox1 = new Box(10, 20, 30);
```

객체의 선언과 생성을 하나의 문장으로 작성할 수 있다.

```java
Box mybox1 = new Box(10, 20, 30);
Avg student1 = new Avg();
String name = new String(“홍길동”);
```

객체는 기본적으로 call by reference를 취한다. 따라서, 선언을 통해 주소를 먼저 할당하고 생성을 통해 주소가 가리키는 곳에 값을 할당한다. (흔히 객체가 생성될 때 메모리에 할당된다고 하는데, 객체가 선언될 때 주소도 메모리에 할당되는 것이 아닐까? 맞다. 그렇지만 주소는 매우 작은 데이터이다.)

반대로, 기본 자료형은 call by value를 취한다. 따라서, 따로 생성을 할 필요가 없다. 기본 자료형을 선언만 한다 하더라도 (예. int st;) 널 값이 저장되기 때문에 객체 측면에서의 생성과 같다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/java_kim/images/2.png" width="70%" height="70%"></p>

**멤버 변수** <br>
클래스에서 속성을 나타내는 변수를 멤버 변수라 한다. 멤버 변수는 클래스에서 메소드 외부에 선언된 변수를 말한다. 멤버 변수는 크게 객체 변수, 클래스 변수, 종단(final) 변수로 구분된다.

`[public/private] [static/final] 변수타입 변수명;`

**객체 변수** <br>
객체 변수는 객체가 가질 수 있는 특성을 표현한다. 객체 변수는 변수가 가지는 값이 기본 자료형(primitive type)(8가지 기본 자료형)의 값인지, 아니면 참조 자료형(reference type)의 값인지에 따라 다른 특성을 가지게 된다. 객체 변수가 가지는 값이 기본 자료형인 경우, 변수가 가지고 있는 것이 값인 반면에, 참조 자료형인 경우 변수가 가지고 있는 것이 값이 아니라 참조 또는 주소이다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/java_kim/images/3.png" width="70%" height="70%"></p>

객체 변수인 my_count1과 my_count2는 서로 다른 기억장소에 저장된 값을 가리킨다. (즉, my_count1의 값이 복사되어 my_count2의 값으로 전달된다)
그러나, 객체 변수인 mybox1과 mybox2는 서로 같은 기억장소에 저장된 값을 가리킨다.  

**클래스 변수(static)** <br>
클래스 변수는 static을 사용하여 선언한다. 클래스 변수는 전역변수(global variable)의 개념을 가진다.

객체 변수는 그 클래스로부터 객체가 생성될 때마다 각 객체에 변수들이 생성되지만, 클래스 변수는 그 클래스로부터 생성된 모든 객체들이 하나의 클래스 변수를 공유한다. 즉, 클래스 변수는 하나의 클래스로부터 생성된 객체들 사이의 통신이나 객체들 사이의 공통되는 속성을 표현하는 데 사용될 수 있다. (static의 역할이 프로그램이 끝날 때까지 살아있음을 표시하는 것이기 때문에 위와 같은 기능을 가질 수 있다)

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/java_kim/images/4.png" width="70%" height="70%"></p>

일반 객체 변수는 객체가 생성될 때마다 메모리에 그 변수의 값을 저장할 수 있는 공간이 생기는 반면, 클래스 변수는 같은 클래스로부터 생성된 모든 객체들이 하나의 클래스 변수값을 공유한다. 이러한 이유로 일반 변수는 객체의 이름을 통해서 접근이 가능하지만, 클래스 변수는 클래스명을 통해서 접근할 수 있다 (물론, 클래스 이름, 객체 이름 모두 접근 가능하다)

```java
class Box3 {
	int width;
	int height;
	int depth;
	long idNum;
	static long boxID = 0;
	public void increment( ) {
		idNum = ++boxID;
	}
}
class Box3Test {
	public static void main(String args[ ]) {
		Box3 mybox1 = new Box3( );
		Box3 mybox2 = new Box3( );
		Box3 mybox3 = new Box3( );
		Box3 mybox4 = new Box4( );
		mybox1.increment( );
		mybox2.increment( );
		mybox3.increment( );
		mybox4.increment( );
		System.out.println(“mybox1의 id 번호: “ + mybox1.idNum);
		System.out.println(“mybox2의 id 번호: “ + mybox1.idNum);
		System.out.println(“mybox3의 id 번호: “ + mybox1.idNum);
		System.out.println(“mybox4의 id 번호: “ + mybox1.idNum);
		System.out.println(“전체 박스의 개수는 “ + Box3.boxID + “입니다.”);   // 이렇게 클래스 변수는 생성된 객체 이름이 아니라 클래스 이름으로 접근하는 게 원칙.
	}
}
```

**종단(final) 변수** <br>
종단 변수는 final을 사용하여 선언하며 변할 수 없는 상수값을 가진다. 즉, final이 붙은 변수는 단 한 번만 초기화할 수 있고 그 이후에는 그 값을 변경할 수 없다. 변수 이름 사용의 관례상 final 변수는 대문자를 사용한다. (C언어에서 define 전처리기와 비슷한 느낌이 난다)

```java
final int MAX = 100;
final int MIN = 1;
```

이처럼 변수 앞에서의 final은 상수값으로 고정시키는 기능을 하고, 클래스 앞에서는 final의 역할은 상속이 안 되도록 한다. 

**멤버 변수 접근 한정자** <br>
자바는 클래스 내의 멤버 변수 접근을 제한할 수 있는 방법으로 접근 한정자를 제공하고 있다. 접근 한정자를 사용한 멤버 변수의 접근 제한은 객체지향 언어의 중요 특성 중에 하나인 캡슐화(encapsulation)와 정보 은폐(information hiding)를 제공한다.

```
[public/private/protected] [static/final] 변수타입 변수명; // 앞에는 접근 한정자이고, 뒤에는 변수 구분자이다.
```

* public으로 선언된 객체 변수는 항상 접근 가능하다. 자바 프로그램에서 반드시 공개해야 되는 경우를 제외하고는 public으로 지정하지 않는 것이 바람직하다. 
* private로 선언된 객체 변수는 소속된 클래스 내에서만 사용할 수 있다. 클래스 외부에서 private로 선언된 객체 변수에 접근하면 오류가 발생한다.
* 객체 변수에 한정자를 사용하지 않고 사용하는 경우가 있는데, 이는 좋은 습관이 아니다. 가능하면 변수의 성격에 따라 한정자를 지정하는 것이 좋은 습관이다. 자바는 한정자를 지정하지 않고 객체 변수를 사용하는 것을 허용한다. (대규모 프로젝트가 아니라면 한정자를 구지 일일이 설정할 필요가 있나?) 한정자를 지정하지 않을 경우에는 같은 패키지에 속한 클래스에서는 제한 없이 사용이 가능하다.

**변수의 유효범위(Scope)** <br>
변수의 유효범위는 그 변수가 사용될 수 있는 영역을 의미한다. 유효범위 측면에서의 변수들을 구분하여 보면 다음과 같이 3가지로 구분할 수 있다.
* 멤버 변수
* 메소드 매개변수와 지역변수
* 예외 처리기 매개변수(exception handler parameter)

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/java_kim/images/5.png" width="70%" height="70%"></p>


## 클래스: 기능

**생성자(Constructor)** <br>
생성자는 클래스로부터 객체가 생성될 때 객체의 초기화 과정을 기술하는 특수한 메소드로 객체가 생성될 때 무조건 수행된다. (생성자는 객체가 생성될 때 한 번만 수행된다) 생성자는 프로그램에 의해 명시적으로 호출되지 않고 객체를 생성하는 new 명령어(예약어)에 의해 자동으로 실행된다. 생성자는 주로 객체 변수를 초기화할 필요가 있을 때 사용하며 생성자의 이름은 반드시 클래스의 이름과 동일해야 한다. 일반적으로 생성자에는 private 한정자를 사용하지 않는다. 생성자는 클래스로부터 객체를 생성할 때 무조건 수행되므로 private으로 지정하게 되면 외부에서 객체를 생성할 때 생성자가 수행될 수 없어 오류가 발생하게 된다. 생성자에 private 한정자를 붙이는 경우는 생성자가 클래스 내부에서만 사용될 때 가능하다.

```java
class Box5 {
	int width;
	int height;
	int depth;
	public Box5(int w, int h, int d) {
		width = w;
		height = h;
		depth = d;
	}
}
```

**생성자 오버로딩(Overloading)** <br>
클래스는 여러 개의 생성자를 가질 수 있다. 여러 개의 생성자를 사용한다는 의미는 같은 이름의 생성자를 여러 개 중첩(overloading)하여 사용할 수 있다는 의미이다. 여러 개의 생성자를 사용할 때 생성자의 이름은 같지만, 생성자가 가지는 매개변수의 타입과 개수는 반드시 달라야 한다. 만일, 한 클래스에 같은 매개변수를 가진 생성자를 2개 이상 사용하면 오류가 발생하게 된다. (생성자 오버로딩이 다형성 특징이다)

```
class Box5 {
	int width;
	int height;
	int depth;
	public Box5() {
		width = 1;
		height = 1;
		depth = 1;
	}
	public Box5(int w) {
		width = w;
		height = 1;
		depth = 1;
	}
	public Box5(int w, int h) {
		width = w;
		height = h;
		depth = 1;
	}
	public Box5(int w, int h, int d) {
		width = w;
		height = h;
		depth = d;
	}
}
public class Box5Test {
	Box mybox1 = new Box5( );
	Box mybox2 = new Box5(10);
	Box mybox3 = new Box5(10,20);
	Box mybox4 = new Box5(10,20,30);
}
```

**예약어 this** <br>
this는 자바의 예약어(reserved word)이며 현재의 객체를 의미한다. 일반적으로 생성자나 메소드의 매개변수가 객체 변수와 같은 이름을 사용하는 경우에 this를 사용하게 된다. (가독성을 위해서라도 this를 사용하는 게 좋지않나?)

```java
class Box5 {
	int width;
	int height;
	int depth;
	public Box5(int width, int height, int depth) {
		this.width = width;
		this.height = height;
		this.depth = depth;
	}
}
```

자바에서 생성자나 메소드의 매개변수 이름이 객체 변수의 이름과 같지 않을 경우에는 this를 사용하지 않아도 된다. 그러나 this의 사용은 객체 변수나 생성자, 메소드의 매개변수 이름을 의미적으로 명확하게 사용할 수 있게 해준다. 의미 있는 변수명을 사용하는 것은 좋은 프로그래밍 작성 요소 중의 하나이다. this의 사용으로 객체 변수나 매개변수의 이름으로 같은 이름을 사용할 수 있다는 이점이 있다.

**메소드** <br>
클래스의 기능에 해당하는 메소드는 객체가 할 수 있는 행동을 정의한 것으로 클래스의 핵심이라 할 수 있다. 

**접근 한정자** <br>
메소드 선언 시 사용되는 접근 한정자는 멤버 변수 접근 한정자와 같이 public, private가 사용된다. 

**클래스 메소드(static)** <br>
클래스 메소드는 클래스 변수와 비슷한 특징을 가진다. 클래스 메소드 역시 클래스 변수처럼 클래스 명과 객체 명을 통해서 접근할 수 있으며, 클래스로부터 생성된 모든 객체들이 공유할 수 있는 메소드이다. 클래스 메소드에는 일반 객체 변수를 사용할 수 없다. 클래스 메소드 내에서는 오직 클래스 변수만이 사용 가능하다. 단, 클래스 메소드 내에서 선언된 지역 변수는 사용할 수 있다 (메모리에 통째로 같이 올라가니까…)

```java
class Box9 {
	private int width;
	private int height;
	private int depth;
	public long idNum;
	static long boxID = 100;
	static long getCurrentID() {
		int count = 1;
		depth++;		// 에러!
		boxID = boxID + count;
		return boxID;
	}
}
…
Box9 mybox1 = new Box9();
System.out.println(“다음 박스의 번호는 “+ Box9.getCurrentID() + “번 입니다”);   // 클래스 메소드 호출할 때 객체 이름이 아닌 클래스 이름을 사용
…
```

**final, abstract, synchronized 메소드** <br>
* final로 선언된 메소드는 서브 클래스에서 오버라이딩(overriding)될 수 없음을 의미한다.
* abstract로 선언된 메소드는 추상 메소드로써 추상 클래스 내에서 선언될 수 있다. 추상 메소드는 선언 부분만 가지고 몸체 부분은 가질 수 없다. 몸체 부분은 서브 클래스에서 오버라이딩된다.
* synchrozied 메소드는 스레드를 동기화할 수 있는 기법을 제공하기 위해 사용된다.

**메소드 반환 값(return value)** <br>
메소드 선언부에는 그 메소드 반환 값의 자료형이 지정되어야 한다. 반환 값이 없을 경우에는 void로 지정한다. void형이 아닌 메소드는 반드시 지정된 형과 같은 값을 return문을 사용하여 반환해야 한다. 메소드는 기본 자료형뿐만 아니라 참조 자료형의 데이터도 반환할 수 있다.

**메소드 오버로딩(overloading)** <br>
생성자의 오버로딩과 같은 개념으로 메소드도 오버로딩될 수 있다. 즉, 같은 클래스에 같은 이름의 메소드를 중첩하여 사용할 수 있다. 물론, 중첩된 메소드들은 매개변수의 형과 개수가 다른 형태를 가져야 한다. 메소드 오버로딩은 객체지향 언어의 특징 중에 하나인 다형성(polymorphism)을 제공한다. 즉, 하나의 메소드 이름으로 다양한 연산을 수행할 수 있는 방법을 제공한다. 중첩된 메소드가 호출되면 매개변수의 형과 개수를 비교하여 적합한 메소드가 실행된다.

**메소드에 값 전달 (argument passing) 방법** <br>
자바에서 메소드 호출 시 매개변수로 지정되는 실 매개변수는 기본 자료형과 참조 자료형으로 나누어 볼 수 있다. 값-전달 방법은 메소드 호출 시 실 매개변수의 값을 형식 매개변수에 복사해 주는 방식이다. 자바의 값-전달 기법은 메소드 호출 시 지정한 실 매개변수의 형에 따라 다르게 작동한다. 즉, 실 매개변수로 기본 자료형을 지정하는 경우와 참조 자료형을 지정하는 경우가 다르게 작동한다. 자바의 8개의 기본 자료형(character, Boolean, byte, short, integer, long, float, double)의 변수에는 실제 그 변수의 값이 저장되어 있다. 이 경우 실 매개변수의 값을 형식 매개변수에 복사해 줌으로서 형식 매개변수의 값이 변해도 실 매개변수의 값은 영향을 받지 않는다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/java_kim/images/6.png" width="70%" height="70%"></p>

8개의 기본 자료형을 제외한 모든 것이 객체 즉 참조 자료형이다. 참조 자료형 변수 즉, 객체 참조 변수가 가지고 있는 값은 실제 값이 아니라 참조 자료형(객체)의 주소이다. 이 경우 실 매개변수의 값을 형식 매개변수에 복사해주면 같은 주소로 취하여 접근하게 된다. 이러한 상황에서 형식 매개변수를 이용하여 객체의 값이 변환되면 실 매개변수를 통한 객체의 값도 변환되게 된다. 즉, 참조 자료형을 함수의 인자로 넘길 때 조심해야 한다. 의도치 않은 값의 변환이 일어날 수 있다.

## 상속

상속(inheritance)은 객체지향 언어의 장점인 모듈의 재사용(reusing)과 코드의 간결성을 제공하는 중요한 특성이다. 자바에서는 상속의 개념을 이용하여 클래스들의 계층 구조를 구성한다. 일반적인 개념들은 이미 클래스로 정의되어 있고 이들을 상속하기만 하여 구체적인 일이나 새로운 일들은 추가적으로 정의해주기만 하면 된다. 이러한 개념은 소프트웨어 모듈의 재사용(reusing) 측면에서의 매우 효율적인 방법이다. (우리는 자바 전문가에 의해 작성된 클래스 라이브러리를 상속하면서 우리가 원하는 구체적인 것들만 구현하기만 하면 된다)

자바에서의 모든 클래스들은 상위(super) 클래스를 가진다. 자바 프로그램에서 사용할 수 있는 클래스 중 최상위 클래스는 java.lang.Object 클래스이다. 즉, 자바 프로그램에서 사용하는 모든 클래스들은 Object 클래스의 하위(sub) 클래스이다. 자바에서 프로그램 작성자가 명시적으로 상속되는 상위 클래스를 지정하지 않으면 묵시적으로 Object 클래스로부터 상속된 것으로 간주한다.

자바에서 클래스 선언 시 상위(super) 클래스를 지정하기 위해 확장을 의미하는 extends라는 예약어를 사용한다. 자바 프로그램에서 클래스의 상속은 상위 클래스의 모든 요소를 상속받고 추가 요소를 더 가지는 확장의 개념이다. 

```
public class Box extends SuperBox {
    …
}
```

멤버 변수의 상속 <br>
클래스가 상속되면 상위 클래스의 멤버 변수들은 접근 한정자에 따라 상속 여부가 결정된다. 

protected 접근 한정자는 같은 패키지 내의 클래스와 같은 패키지는 아니지만 상속된 클래스에서 사용 가능한 접근 한정자이다. (상속이라는 개념 때문에 protected 키워드가 등장한 셈이다)

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/java_kim/images/7.png" width="70%" height="70%"></p>

멤버 변수의 접근 한정자는 다음과 같이 정리된다.
* public: 동일한 패키지인지, 상속관계인지에 상관없이 모든 클래스에서 사용 가능
* 한정자없음: 동일한 패키지이면 상속 여부에 상관없이 사용 가능
* protected: 동일한 패키지이면 상속 여부에 상관없이 사용 가능하며, 다른 패키지라도 상속되었으면 사용 가능
* private: 어떠한 경우에도 사용 불가능. 클래스 내부에서만 사용 가능

**메소드의 상속과 오버라이딩(overriding)** <br>
클래스들의 상속관계에서 상위 클래스에 선언된 메소드는 접근 한정자에 따라 상속 여부가 결정된다. 메소드에 부여된 접근 한정자의 의미는 객체 변수에 부여된 의미와 동일하다. 당연히 private로 선언된 메소드는 하위 클래스에 상속되지 않는다. 

오버로딩(overloading)은 같은 클래스 내에서 같은 이름의 생성자나 메소드를 사용하는 경우이다. 오버라이딩(overriding)은 상속 관계에 있는 클래스들 간에 같은 이름의 메소드를 정의하는 경우이다. 오버라이딩은 상위 클래스의 메소드와 하위 클래스의 메소드가 메소드 이름은 물론 매개변수의 타입과 개수까지도 같아야 한다. (오버로딩은 같은 이름의 메소드들을 여러 개 사용하려는 의도, 오버라이딩은 기존에 상위 레벨에 있던 메소드를 내가 새롭게 정의하고 싶은 의도)

자바에서 오버로딩과 오버라이딩을 사용하는 이유는 객체지향 언어의 주요 개념인 다형성(polymorphism)을 제공하기 위함이다. 오버로딩(overloading)은 같은 클래스 내에서 같은 이름의 메소드를 정의하여 다형성을 지원하고, 오버라이딩(overriding)은 상속관계에 있는 상위 클래스와 하위 클래스에서 같은 이름의 메소드를 정의하여 다형성을 지원한다. 기존의 클래스를 이용하여 새로운 클래스를 만들 때, 기존 클래스의 메소드와 의미적으로는 같지만 구현 부분에서 약간의 변화가 필요하다면 메소드 오버라이딩을 이용하여 새로운 클래스를 작성할 수 있다. 

상속관계의 클래스에서 메소드가 오버라이딩되었다면 상위 클래스의 메소드가 하위 클래스에 의해 가려지게 된다 (이게 오버라이딩의 목표임). 하위 클래스의 객체에서 상위 클래스에서 오버라이딩된 메소드를 사용하려면 예약어 super를 이용해야 한다.

다음은 오버라이딩이 되지 않는 예이다. 상위 클래스와 하위 클래스의 메소드 이름은 같지만 매개변수의 개수와 타입이 같지 않아 오버라이딩이 되지 않는다. 이 경우 하위 클래스의 입장에서 보면 메소드가 오버로딩된 것으로 볼 수 있다. (이처럼 오버로딩과 오버라이딩은 밀접한 관계를 가진다)

```java
class Da {
	void show(String str) {
		System.out.println(“상위 클래스의 메소드(string) 수행 “ +str);
	}
}
class Db extends Da {
	void show( ) {
		System.out.println(“하위클래스의 메소드 수행”);
	}
}
public class OverridingTest1 {
	public static void main(String args[ ]) {
		Db over = new Db( );
		over.show(“ – 자바 안녕”)	     // Da의 메소드가 실행
		over.show( ):;			// Db의 메소드가 실행
	}
}
```

다음은 상위 클래스에서 선언된 메소드를 오버라이딩하는 예이다. 상위 클래스의 메소드는 하위 클래스에 상속되지 못하고 가려지게 된다 (성공!)

```java
class Ea {
	void show( ) {
		System.out.println(“상위 클래스의 메소드(string) 수행 “ +str);
	}
}
class Eb extends Ea {
	void show( ) {
		System.out.println(“하위클래스의 메소드 수행”);
	}
}
public class OverridingTest1 {
	public static void main(String args[ ]) {
		Eb over = new Eb( );
		over.show( );	// Eb의 메소드가 실행
	}
}
```

**예약어 super** <br>
예약어 super는 두 가지 형태로 사용된다. 
* 첫째는 하위 클래스에 의해 가려진 상위 클래스의 멤버 변수나 메소드에 접근할 때 사용
* 둘째는 상위 클래스의 생성자를 호출하기 위해 사용

this: 현재 객체의 주소 <br>
super: 상위 클래스 객체의 주소

**상속과 생성자** <br>
클래스에서 생성자는 객체가 생성될 때 초기화 역할을 수행한다. 생성자 중에 매개변수가 없는 생성자를 묵시적(default) 생성자라 한다. 클래스가 상속관계에 있을 때 각 클래스들이 묵시적 생성자를 모두 가지고 있다면, 하위 클래스에서 객체가 생성될 때 상위 클래스의 묵시적 생성자가 하위 클래스의 묵시적 생성자보다 먼저 자동으로 수행된다.

묵시적 생성자가 아닌 매개변수가 있는 생성자의 경우에는 명시적으로 상위 클래스의 생성자를 호출해주어야 수행된다. 상위 클래스의 생성자를 명시적으로 호출하기 위해 예약어 super를 사용한다. (상위 클래스의 매개변수와 하위 클래스의 배개변수의 개수나 타입이 다를 때 명시적으로 super를 사용해서 생성자를 호출해줘야 한다)

주의할 점은 상위 클래스의 특정 생성자를 호출하는 super 문장은 반드시 생성자 부분의 첫 번째 라인에 위치해야 한다. 이것은 상위 클래스의 생성자가 항상 하위 클래스 생성자보다 먼저 수행되어야 함을 의미한다.

```java
class Ad1 {
	int d1;
	int s;
	Ad1(int s1) {
		System.out.println(“클래스 Ad1의 생성자 수행”);
		s = s1;
		d1 = s * s;
	}
}
class Ad2 extends Ad1 {
	int d2;
	int t;
	Ad2(int s1, int t1) {
		super(s1);	// 상위 클래스 생성자 호출
		System.out.println(“클래스 Ad2의 생성자 수행”);
		t = t1;
		d2 = t * t;
	}
}
public class SuperTest3 {
	public static void main(String args[ ]) {
		Ad2 super2 = new Ad2(10, 20);
		System.out.println(“10의 제곱은: “ + super2.d1);
		System.out.println(“20의 제곱은: “ + super2.d2);
	}
}
```

**객체의 형변환** <br>
자바는 클래스 계층 구조에서 상속관계의 클래스로부터 생성된 객체들 사이의 형변환을 허용한다.
* 하위 클래스에서 생성된 객체를 상위 클래스 형의 객체 변수에 배정하는 형변환은 허용
* 반대로 상위 클래스에서 생성된 객체를 하위 클래스 형의 객체 변수에 배정할 수 없음
* 상위 클래스 형의 객체 변수에 배정된 하위 클래스 객체의 경우, 상위 클래스 형의 객체 변수를 통해서는 상위 클래스에 선언된 속성에만 접근이 가능

```java
class Am {
	void callme( ) {		// 3개의 상속된 클래스에 메소드 오버라이딩
		System.out.println(“클래스 Am의 callme() 메소드 실행”);
	}
}
class Bm extends Am {
	void callme( ) {		// 3개의 상속된 클래스에 메소드 오버라이딩
		System.out.println(“클래스 Bm의 callme() 메소드 실행”);
	}
}
class Cm extends Am {
	void callme( ) {		// 3개의 상속된 클래스에 메소드 오버라이딩
		System.out.println(“클래스 Cm의 callme() 메소드 실행”);
	}
}
public class OverridingAndCasting {
	public static void main(String args[ ]) {
		Am r = new Am( );	// Am 클래스의 객체 변수 r에 Am 클래스 객체 배정
		r.callme( );
		r = new Bm( );		// Am 클래스의 객체 변수 r에 Bm 클래스 객체 배정
		r.callme( );
		r = new Cm( );		// Am 클래스의 객체 변수 r에 Cm 클래스 객체 배정
		r.callme( );
	}
}
```
자바에서는 오버라이딩된 메소드의 선정(binding)을 실행시간에 수행(dynamic method dispatch)하고 있다. 자바는 객체의 형변환과 메소드 오버라이딩을 이용하여 객체지향 언어의 다형성(polymorphism)을 제공하고 있다.

**연산자 instanceof** <br>
이 연산자는 객체가 특정 클래스나 인터페이스로부터 생성된 객체인지를 판별하여 true 또는 false 값을 반환해주는 이진 연산자이다.
```java
if(oba instanceof String)
	System.out.println(“oba는 String 클래스의 객체입니다”);
```

**추상 클래스와 추상 메소드** <br>
추상 클래스는 객체지향 언어의 중요한 개념이다.
추상 클래스는 하위 클래스에서 구현되는 추상적인 기능만을 정의하는 클래스이다.
추상 클래스에서 정의된 추상적인 기능은 하위 클래스에서 구현된다.
(전문가인 내가 틀을 잡아놓을 테니 너네들은 주어진 틀 안에서 구현을 완성해봐라…)

추상 클래스는 기능이 무엇(what)인지만을 정의하고 어떻게(how) 구현되는지는 정의하지 않는다.
추상 클래스에서 선언된 기능이 하위 클래스에서 구현되므로, 하나의 추상 클래스에 정의된 기능을 여러 개의 하위 클래스에서 서로 다른 형태로 구현하여 사용할 수 있다.

추상 메소드는 추상 클래스 내에 정의되는 메소드로써 선언 부분만 있고 구현 부분이 없는 메소드이다. 추상 클래스를 상위 클래스로 하는 하위 클래스에서는 상위 클래스의 추상 메소드를 서로 다른 방법으로 구현하여 사용할 수 있다. 즉, 하위 클래스에서는 상위 클래스에서 추상 메소드로 정의된 메소드를 오버라이딩해서 사용한다.

이러한 추상 메소드의 사용은 객체지향 언어에서의 다형성을 지원하는 요소이다. 추상 메소드로 정의된 메소드 이름은 하위 클래스에서 공통적으로 사용될 수 있으며, 서로 다르게 구현될 수 있다. 하나의 메소드 이름을 사용하지만 서로 다른 구현 방법을 가질 수 있다.

추상 클래스는 최소한 하나 이상의 추상 메소드를 가져야 하며, 추상 클래스로부터는 직접 객체가 생성될 수 없다. 왜냐하면 추상 메소드는 구현 부분이 없는 메소드이기 때문이다. (추상 클래스의 다형성은 오로지 오버라이딩밖에 될 수 없다? 자기 자신은 없으니 선택되지 못해 오버로딩은 )

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/java_kim/images/8.png" width="70%" height="70%"></p>

```java
abstract class Shape {	// 추상 클래스와 추상 메소드 선언
	abstract void draw( );
	abstract void computeArea(double a, double b);
}
class Circle extends Shape {
	void draw( ) {
		System.out.println(“원을 그리는 기능”);
	}
	void computeArea(double r1, double r2) {
		System.out.println(“원의 넓이: “ + (3.14 * r1 * r2));
	}
}
class Rectangle extends Shape {
	void draw( ) {
		System.out.println(“사각형을 그리는 기능”);
	}
	void computeArea(double h, double v) {
		System.out.println(“사각형의 넓이: “ + (h * v));
	}
}
class Triangle extends Shape {
	void draw( ) {
		System.out.println(“삼각형을 그리는 기능”);
	}
	void computeArea(double a, double h) {
		System.out.println(“삼각형의 넓이: “ + (a * h / 2));
	}
}
public class AbstractTest {
	public static void main(String args[ ]) {
		// 객체 형변환과 오버라이딩을 이용
		Shape s = new Circle( );
		s.draw( );
		s.computeArea(5.0, 10.0);
		s = new Rectangle( );
		s.draw( );
		s.computeArea(5.0, 5.0);
		s = new Triangle( );
		s.draw( );
		s.computeArea(5.0, 10.0);
	}
}
```

**예약어 final** <br>
final은 자바에서 3가지 기능이 있다
* 객체 변수에 final을 붙이면, 상수로 사용된다
* 메소드에 final을 붙이면, 하위 클래스에서 오버라이딩하여 사용할 수 없다
* 클래스에 final을 붙이면, 상속을 허용하지 않는다

자바에서 final을 사용하는 이유는 보안을 명확하게 하기 위해서이다. 
해커들이 시스템에 침입하는 방법 중의 하나가 기존의 클래스를 이용하여 하위 클래스를 작성한 다음 하위 클래스를 상위 클래스로 대치시키는 방법이다. 하위 클래스에 해킹을 가하는 부분을 추가할 수 있기 때문이다.

자바에서 final이 필요한 이유는 설계 부분 때문이다. 만일 클래스가 개념적으로 완벽하게 구현되어 있다면 그 클래스는 더 이상 하위 클래스를 가질 필요가 없기 때문에 final로 선언한다.

## 인터페이스와 예외처리

인터페이스(interface)는 상수와 메소드 선언들의 집합이다. 인터페이스는 추상 클래스보다 더욱 완벽한 추상화를 제공한다. 추상 클래스는 추상 메소드 외에 다른 멤버 변수나 일반 메소드를 가질 수 있지만, 인터페이스는 추상 메소드(메소드 선언만 있는)와 상수만을 가진다. 즉, 인터페이스는 추상 메소드와 상수만으로 구성된 추상 클래스라 할 수 있다.

그러면 왜 인터페이스를 사용하는가? <br>
자바에서의 클래스는 상위 클래스(상속받는 클래스)로 하나의 클래스만 지정할 수 있다. 즉, 자바는 다중 상속(multiple inheritance)을 지원하지 않는다. 그러나 자바는 완벽한 다중 상속의 개념은 아니지만, 인터페이스를 사용함으로써 다중 상속을 흉내 낼 수 있다.

인터페이스는 추상 클래스는 비슷한 특성을 가진다. 응용 프로그램의 특성에 따라 어느 것을 사용할 지 결정할 수 있다. 현재의 클래스가 이미 다른 클래스로부터 상속을 받고 있는 상태이면서, 또 다른 클래스의 요소들이 필요하다면 이 때는 인터페이스를 사용해야 한다.

자바의 개발 도구인 JDK에는 클래스 라이브러리뿐만 아니라 400개 이상의 많은 인터페이스들이 제공되고 있다. 사용자가 자바 프로그램을 개발할 때 이러한 인터페이스를 사용한다.

인터페이스와 인터페이스에 정의된 메소드의 접근 한정자는 public만을 사용해야 한다. 왜냐하면 상속받은 클래스에서 사용되어야 하기 때문이다. 메소드는 구현 부분 없이 선언 부분만 정의되어야 한다.

**인터페이스의 사용** <br>
클래스에서 인터페이스를 사용하기 위해서는 implements 예약어를 사용한다. 클래스에서 인터페이스를 사용할 경우 인터페이스에서 정의된 모든 메소드를 클래스 내에서 반드시 오버라이딩하여 구현해야 한다.

```
[public/final/abstract] class 클래스이름 extends 상위클래스이름 implements 인터페이스이름(들)
{
… 멤버 변수 선언
… 생성자
… 메소드 선언
… 인터페이스에 선언된 모든 메소드를 오버라이딩하여 구현해야 한다.
}
```

```java
public interface Sleeper {
	public long ONE_SECOND = 1000;
	public long ONE_MINUTE = 60000;
	public void wakeup();
}
public interface Worker {
	public long WORK_TIME = 8;
	public void sleep();
}
// 위의 interface는 보통 자바 전문가가 미리 구현해 놓았을 것이다.
public class Man implements Sleeper, Worker {	// 이 클래스가 개발자가 구현하는 부분
	public void wakeup() {
		System.out.println(“빨리 일어나”);
	}
	public void sleep() {
		System.out.println(“빨리 자”);
	}
}
```

**인터페이스 상속** <br>
인터페이스 선언 시 필요에 따라 다른 인터페이스로부터 상속을 받을 수 있다. 인터페이스들의 상속에도 extends 예약어를 사용한다. 

```
public interface 인터페이스이름 extends 인터페이스이름(들) {
… 상수 선언
… 메소드 선언
}
```

인터페이스를 상속받을 클래스는 인터페이스의 모든 메소드가 구현되어야 한다.

**예외의 개요**
 예외 처리는 예를 들어,
* 정수를 0으로 나누는 경우
* 배열의 첨자가 음수 값을 가지는 경우
* 배열의 첨자가 배열의 크기를 벗어나는 경우
* 부적절한 형변환이 일어나는 경우
* 입출력 시 인터럽트가 발생하는 경우
* 입출력을 위해 지정한 파일이 없거나 파일 이름이 틀린 경우

등이 발생하면 처리되는 루틴이다. 자바에서는 발생되는 모든 예외를 객체로 취급하고 있으며 자바의 JDK에서는 예외 관련 클래스들을 제공한다. JVM은 프로그램 실행 중에 예외가 발생하면 관련된 예외 클래스로부터 예외 객체를 생성하여 프로그램에서 지정된 예외 처리 루틴에 넘겨준다. 프로그램에 지정된 예외 처리 루틴은 예외 발생 시 JVM에 의해 호출되며 예외 객체를 JVM으로부터 넘겨받아 적절한 처리를 수행한다. 예외 처리 루틴이 없는 프로그램을 수행 중에 예외가 발생하면 JVM은 default 예외 처리기를 작동시킨다. 

자바는 RuntimeException과 Error 클래스(예외 처리 루틴이 매우 복잡해서 JVM이 대신 자동으로 해줌)를 제외 한 나머지 클래스와 관련된 예외는 개발자가 프로그램에서 직접 처리하도록 요구한다. 예를 들어, 사용자가 입출력 관련 예외 처리를 지정하지 않으면 자바 컴파일러는 오류를 발생시킨다. 

**메소드에서 예외 처리** <br>
발생되는 예외를 메소드 내에서 직접 처리하기 위해 자바 언어는 try, catch, finally 구문을 제공한다.
```java
import java.io.*
public class ExceptionTest1 {
    public static void main(String args[]) {
        try {
	    FileReader file = new FileReader("a.txt");
	    int i;
	    while((i = file.read()) != -1)
	        System.out.print((char)i);
	    file.close();
	}
	catch (Exception e) {
	    System.out.println("예외 처리 루틴: " + e + " 파일이 존재하지 않는다.");
	}
    }
}
```

* try 블록: 조건 없이 실행되는 문장. 예외가 발생할 가능성이 있는 범위를 지정. try 블록은 적어도 하나 이상의 catch 블록을 가져야 함.
* catch 블록: catch 블록의 매개변수는 예외 객체로서 예외에 관한 정보를 저장한다. catch 블록의 매개변수로 지정되는 객체의 타입은 반드시 java.lang.Throwable 클래스의 하위 클래스이어야 한다. try 블록에서 다양한 예외가 발생할 수 있으며, 이러한 예외의 종류에 따라 여러 개의 catch 블록이 지정될 수 있다.
* fianlly 블록: 필요에 따라 선택적으로 사용됨. 예외의 발생과 상관없이 무조건 수행되는 블록. 자바 응용 프로그램에서 finally 블록은 파일이나 데이터베이스를 닫는 마무리 기능으로 많이 활용된다.

**호출한 메소드에 예외를 넘겨주는 방법** <br>
발생된 예외를 메소드 내에서 처리하지 않고, 호출한 메소드에 넘겨주는 방법이다. 이러한 방법은 처리해야 하는 모든 예외를 하나의 메소드에 처리하게 하거나, 자바 가상 기계(JVM)에 처리를 맡길 때 유용하다. (RuntimeException과 Error 클래스처럼 거의 예외 처리를 자동으로 처리하는 것과 비슷하다)

```java
import java.io.*;
public class ExceptionTest2 {
    public static void main(String args[]) throws Exception { // throws() 절을 이용하여 예외를 JVM에 넘김.
        FileReader file = new FileReader("a.txt");
	int i;
	while((i = file.read()) != -1)
	    System.out.print((char)i);
	file.close();
    }
}
```

## 입출력

입출력은 다양한 형태의 입출력 장치들과 연관되어 있기 때문에 어떤 언어에서나 처리하기 간단하지 않다. 자바는 특정 하드웨어에 종속되지 않도록 설계된 언어이므로 어떤 컴퓨터에서나 일관된 형태로 사용할 수 있도록 제공한다. 

자바는 입출력을 위해 스트림(stream)을 사용한다. 스트림이란 순서가 있는 일련의 데이터를 의미하는 추상적인 개념이다. 사용자가 프로그램에서 스트림을 이용하여 입출력을 수행하면, 입출력 스트림을 실제 하드웨어 장치에 연결하는 것은 JVM에 의해 이루어진다. 

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/java_kim/images/9.PNG" width="70%" height="70%"></p>

**파일과 디렉터리** <br>
File 클래스는 입출력을 위해 필요한 파일과 디렉터리를 다룬다. 예제를 살펴보자.
```java
import java.io.File;
public class FileDirTest1 {
    public static void main(String args[]) {
        String directory = "C:\Windows";
	File f1 = new File(directory);
	if (f1.isDirectory()) {
	    System.out.println("검색 디렉터리 " + directory);
	    System.out.println("==============================");
	    String s[] = f1.list();    // 디렉터리에 있는 모든 요소를 문자열 배열로 생성
	    for (int i=0; i < s.length; i++) {
	        File f = new File(directory + "/" + s[i]);   // 각 요소를 File 객체로 생성
		if (f.isDirectory())
		    System.out.println(s[i] + " : 디렉터리");
		else
		    System.out.println(s[i] + " : 파일");
	    }
	}
	else
	    System.out.println("지정한 " + directory + " 는 디렉터리가 아님");
    }
}
```

**문자(Character) 스트림과 바이트(Byte) 스트림** <br>
스트림은 입출력 데이터의 추상적인 표현이다. 즉, 사용자는 스트림을 이용하여 실제 하드웨어 장치와 상관없이 일관된 입출력 방법을 사용할 수 있다. 예를 들어, 입력 스트림은 키보드, 파일, 메모리 버퍼, 포트 등으로부터 입력되는 데이터의 일관된 표현이고, 출력 스트림은 모니터, 파일, 메모리 버퍼, 포트 등에 출력되는 데이터의 일관된 표현이다. 즉, 어떤 입출력 장치에서 입출력이 수행되더라도 사용자는 스트림 형태만 기억하고 있으면 된다.

스트림에는 문자 스트림과 바이트 스트림 두 가지 형태가 있다. 
* 문자 스트림: 16 비트 문자나 문자열들을 읽고 쓰기 위한 스트림
   * 문자 스트림의 입출력을 위해서 Reader, Writer 클래스와 그 하위 클래스를 이용
* 바이트 스트림(바이너리 스트림): 8비트의 바이트를 읽고 쓰기 위한 스트림
   * 바이트 스트림의 입출력을 위해서는 InputStream, OutputStream 클래스와 그 하위 클래스를 이용



p403부터 시작

