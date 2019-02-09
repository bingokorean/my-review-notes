# Week 1. Programming Challenges

Learning Objectives
* Practice implementing algorithms
* Practice testing and stress testing programs
* Compare fast and slow programs
* Practice solving programming challenges

## Welcome

## Programming Assignment 1: Programming Challenges

Programming challenge in five steps:
1. Reading proglram statement
   * input/output format 확인
   * contraints for input, time, memory limits 확인
2. Designing an algorithm
   * start designing an algorithm, and prove that it works correctly
   * running time에 특히 더 신경써야 함
3. Implementing an algorithm
   * start implementing it in a programming language 
   * 코딩하기 전에 수도코드로 먼저 짜볼 것, 그리고 starter solution부터 먼저 짜보자 (나중에 stress testing에 쓰일 수도 있음).
4. Testing and debugging your program
   * testing - revealing bugs
   * debugging - exterminating the bugs
   * program is ready -> test it -> bug found -> fix it -> test it -> ... again.
5. Submitting your program to the grading system


### 1. Sum of Two Digits

### 2. Maximum Pairwise Product

#### 2.1 Naive Algorithm
go through all possible pairs of the input elements
```
MaxPairwiseProductNaive(A[1...n]): 
product←0 for i from 1 to n: 
for j from 1 to n: if i ,j: 
    if product < A[i]·A[j]: 
        product ← A[i]·A[j]
return product
```
This code cab be optimized and made more compact as follows:

```
# Uses C++ 
#include <iostream> 
#include <vector>

using std::vector; 
using std::cin; 
using std::cout; 
using std::max;

int MaxPairwiseProduct(const vector<int>& numbers) { 
    int product = 0; 
    int n = numbers.size(); 
    for (int i = 0; i < n; ++i) { 
        for (int j = i + 1; j < n; ++j) { 
            product = max(product, numbers[i] * numbers[j]); 
        } 
    } 
    return product; 
}

int main() { 
    int n; 
    cin >> n; 
    vector<int> numbers(n); 
    for (int i = 0; i < n; ++i) { 
        cin >> numbers[i]; 
    }
    
    int product = MaxPairwiseProduct(numbers); 
    cout << product << "\n"; 
    return 0;
}

```

```
# Uses Java 
import java.util.*; 
import java.io.*;

public class MaxPairwiseProduct {

    static int getMaxPairwiseProduct(int[] numbers) {
        int product = 0;
        int n = numbers.length;
        for (int i=0; i<n; ++i) {
            for (int j=i+1; j<n; ++j) {
                product = Math.max(product, numbers[i] * numbers[i]);
            }
        }
        return product;
    }
    
    public static void main(String[] args) {
        FastScanner scanner = new FastScanner(System.in);
        int n = scanner.nextInt();
        int[] numbers = new int[n];
        for (int i=0; i<n; i++) {
            numbers[i] = scanner.nextInt();
        }
        System.out.println(getMaxPairwiseProduct(numbers));
    }
    
    static class FastScanner {
        BufferedReader br;
        StringTokenizer st; 
        
        FastScanner(InputStream stream) {
            try {
                br = new BufferedReader(new InputStreamReader(stream));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        
        String next() {
            while (st == null || !st.hasMoreTokens()) {
                try {
                    st = new StringTokenizer(br.readLine());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            return st.nextToken();
        }
        
        int nextInt() {
            return Integer.parseInt(next());
        }
    }
}
```

```
# Uses python3 
n = int(input()) 
a = [int(x) for x in input().split()]

product = 0

for i in range(n): 
    for j in range(i + 1, n): 
        product = max(product, a[i] * a[j])
        
print(product)
```

그러나.. 이 알고리즘은 테스트에 통과하지 못한다. 입력 시퀀스 길이 n에 대해서, n의 제곱에 해당하는 time complexity가 발생하기 때문이다. 

#### 2.2 Fast Algorithm

더 빠른 방법이 없을까? 일일이 루프를 돌면서 곱셈을 하는 것보다 largest element와 the second largest element를 찾기만 하면 되지 않는가? 이렇게하면 time complexity가 n 제곱이 아닌 2n이 된다.

```
MaxPairwiseProductFast(A[1...n]):
index_1 ← 1
for i from 2 to n:
    if A[i] > A[index_1]:
        index_1 ← i
index_2 ← 1     # 이 부분이 좀...
for i from 2 to n:
    if A[i] is not A[index_1] and A[i] > A[index_2]:
        index_2 ← i
 return A[index_1] * A[index_2]
```

#### 2.3 Testing and Debugging

테스팅과 디버깅을 통해 다음과 같은 보완된 코드를 얻을 수 있다. 입력 A=[2,1]인 경우. index_1과 index_2가 동일하게 1이 되는 문제점이 있다. 이를 보완하기 위해 index_1이 1일 경우 index_2는 다른 값을 초기화하도록 해야한다.

```
MaxPairwiseProductFast(A[1...n]):
index_1 ← 1
for i from 2 to n:
    if A[i] > A[index_1]:
        index_1 ← i
if index_1 = 1:    # 이렇게 예외처리를 해줘야 한다.
    index_2 = 2
else
    index_2 ← 1
for i from 2 to n:
    if A[i] is not A[index_1] and A[i] > A[index_2]:
        index_2 ← i
 return A[index_1] * A[index_2]
```

이제 A=[100000, 90000] 로 테스트해보자. 올바른 답이 나오지 않는가? 아마 integer overflow 문제일 것이다. c++이라면 int32_t type이 아닌 int64_t type을 사용하자. 

더 많은 케이스들을 테스트하고 싶으면 dataset.txt와 같은 파일 입출력으로 하자.

#### 2.4 Can You Tell Me What Error Have I Made?

> Failed case #5/17: wrong answer 

5번째 케이스가 왜 틀렸는지 알 수 없다. 아무도 테스트 케이스를 보여주진 않는다. 숙련된 프로그래머라 하더라도 실수를 하기 때문에 버그는 최대한 빨리 발견할 수록 좋다. 몇 개의 테스트 케이스만으로 자신이 작성한 프로그램에 신뢰를 가지지 말자. 시점에 따라 버그는 쉽게 발견될 수 있고 매우 어렵게 발견될 수 있다. 따라서, 처음부터 면밀하게 테스트와 디버깅을 해야한다. 

> Learning how to implement algorithms as well as test and debug your programs will be invaluable in your future work as a programmer.

#### 2.5 Stress Testing

stress testing이란 작성된 프로그램의 오류를 찾을 수 있도록 수천개의 테스트를 자동적으로 만드는 방법이다. 다음과 같이 4개의 part를 가진다.
* Your implementation of an algorithm
* An alternative, trivial and slow, but correct implementatino of an algorithm for the same problem
* A random test generator
* An infinite loop, where a new test is generated and fed into both implementations to compare the results

즉, 아주 느리더라도 정확한 알고리즘의 출력을 정답으로 간주하고 테스트 케이스를 만든다. 

MaxPairwiseProductFast에 대한 stress test를 실시해보자.

```
StressTest(N, M):
while true:
    n ← random integer between 2 and N
    allocate array A[1...n]
    for i from 1 to n:
       A[i] ← random integer between 0 and M
    print(A[1...n])
    result_1 ← MaxPairwiseProductNaive(A)
    result_2 ← MaxPairwiseProductFast(A)
    if result_1 = result_2:
        print("OK")
    else:
        print("Wrong answer: ", result_1, result_2)
        return
```

Wrong answer이 안나오길 기대하며 테스팅(testing)을 해보자.

```
67232 68874 69499 
OK 
6132 56210 45236 95361 68380 16906 80495 95298 
OK 
62180 1856 89047 14251 8362 34171 93584 87362 83341 8784 
OK 
21468 16859 82178 70496 82939 44491
OK 
68165 87637 74297 2904 32873 86010 87637 66131 82858 82935 
Wrong answer: 7680243769 7537658370 
```

틀린 케이스를 찾았다. 이제 버그를 찾기 위한 디버깅(debugging)을 진행하자. 테스트는 쉽지만 디버깅은 무척 어렵다. 디버깅을 하기 전에 테스트 케이스를 좀 단순화시켜보자.

```
... 
7 3 6 
OK 
2 9 3 1 9 
Wrong answer: 81 27
```
MaxPairwiseProductFast(A) 가 틀린 것으로 확인된다. 

```
MaxPairwiseProductFast(A[1...n]):
index_1 ← 1
for i from 2 to n:
    if A[i] > A[index_1]:
        index_1 ← i
if index_1 = 1:    # 이렇게 예외처리를 해줘야 한다.
    index_2 = 2
else
    index_2 ← 1
for i from 2 to n:
    if i is not index_1 and A[i] > A[index_2]:    # A[i] is not A[index1] 이 아니라 i is not index_1 로 해야 한다.
        index_2 ← i
 return A[index_1] * A[index_2]
```

다음 psudocode는 more "reliable"한 방식의 코드이다. swap을 통해서 코드를 훨씬 간결하게 만든다.

```
MaxPairwiseProductFast(A[1...n]):
index ← 1
for i from 2 to n:
    if A[i] > A[index_1]:
        index_1 ← i
swap A[index] and A[n]
index ← 1
for i from 2 to n-1:        # n-1
    if A[i] > A[index]:
        index ← i
swap A[index] and A[n-1]
return A[n-1] * A[n]
```

#### 2.6 Even Faster Algorithm
 MaxPairwiseProductFast 알고리즘은 대략 2n time complexity를 가지며 largest와 second largest element를 찾는다. 하나의 array에서 두 개의 largest element를 가지는 방식은 대략 1.5n comparison을 가진다. n+[log-2 n]-2 comparison 이하는 절대로 가질 수 없을 것이다. 수학적으로 증명되는 부분인 것 같다. 

#### 2.7 A MOre Compact Algorithm
sorting (non-decreasing order)을 활용하면 compact한 알고리즘을 얻을 수 있다.
```
MaxPairwiseProductBySorting(A[1...n]): 
Sort(A) 
return A[n−1]·A[n] 
```
위 알고리즘의 running time은 O(nlogn) 이다 (not O(n)). 

---


APlusB.py
```
def sum_of_two_digits(first_digit, second_digit):
    return first_digit + second_digit

if __name__ == '__main__':
    a, b = map(int, input().split())
    print(sum_of_two_digits(a, b))
```


max_pairwise_product.py

```
def max_pairwise_product(numbers):
    n = len(numbers)
    numbers.sort()
    return numbers[-1] * numbers[-2]

if __name__ == '__main__':
    input_n = int(input())
    input_numbers = [int(x) for x in input().split()]
    print(max_pairwise_product(input_numbers))
```
