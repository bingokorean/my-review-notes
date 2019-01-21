# Week 1. Programming Challenges

## Welcome

## Programming Assignment 1: Programming Challenges

Programming challenge in five steps:
1. Reading proglram statement
   * input/output format 확인
   * contraints for input, time, memory limits 확인
2. Designing an algorithm
   * start designing an algorithm, and prove that it works correctly
3. Implementing an algorithm
   * start implementing it in a programming language (코딩하기 전에 수도코드로 먼저 짜볼 것)
4. Testing and debugging your program
   * testing - revealing bugs
   * debugging - exterminating the bugs
   * program is ready -> test it -> but found -> fix it -> text it -> ... again.
5. Submitting your program to the grading system


### 1. Sum of Two Digits

### 2. Maximum Pairwise Product

#### 2.1 Naive Algorithm
go through all possible pairs of the input elements
```
MaxPairwiseProductNaive(A[1...n]): 
product←0 for i from 1 to n: 
for j from 1 to n: if i ,j: 
    if product< A[i]·A[j]: 
        product←A[i]·A[j]
return product
```
This code cab be optimized and made more compact as follows:

```c++
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

```java
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

```python
# Uses python3 
n = int(input()) 
a = [int(x) for x in input().split()]

product = 0

for i in range(n): 
    for j in range(i + 1, n): 
        product = max(product, a[i] * a[j])
        
print(product
```

그러나.. 이 알고리즘은 테스트에 통과하지 못한다. 입력 시퀀스 길이 n에 대해서, n의 제곱에 해당하는 time complexity가 발생하기 때문이다. 

#### 2.2 Fast Algorithm

더 빠른 방법이 없을까? 일일이 루프를 돌면서 곱셈을 하는 것보다 largest element와 the second largest element를 찾기만 하면 되지 않는가? 이렇게하면 time complexity가 n 제곱이 아닌 2n이 된다.

```
MaxPairwiseProductFast(A[1...n]):
index_1 <- 1
for i from 2 to n:
    if A[i] > A[index_1]:
        index_1 <- i
index_2 <- 1
for i from 2 to n:
    if A[i] is not A[index_1] and A[i] > A[index_2]:
        index_2 <- i
 return A[index_1] * A[index_2]
```

