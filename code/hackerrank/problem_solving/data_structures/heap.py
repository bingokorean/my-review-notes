###################################################################################
""" QHEAP1 """
###################################################################################
''' by dcod5, Easy, 25, https://www.hackerrank.com/challenges/qheap1/problem
Heap 자료구조를 활용해서 다음 명령들을 수행하는 알고리즘을 설계하시오.
 - "1 v" - Add an element  to the heap.
 - "2 v" - Delete the element  from the heap.
 - "3" - Print the minimum of all the elements in the heap.
NOTE: It is guaranteed that the element to be deleted will be there in the heap. Also, at any instant, only distinct elements will be in the heap.
### Sample Input
5  
1 4  
1 9  
3  
2 4  
3  
### Sample Output
4  
9 
### Explanation
After the first 2 queries, the heap contains {4,9}. Printing the minimum gives 4 as the output. Then, the 4th query deletes 4 from the heap, and the 5th query gives 9 as the output.
'''

import heapq
# heap_st_li 내부의 첫 번째 index의 값을 삭제하면 heap structure가 흐트러진다. 
# 따라서, 첫 번째 index 값을 삭제한 후, heapify를 통해 heap structure를 재구성한다.
# 중간중간에 heap_st_li 를 관찰해보면 정렬되어 있진 않지만 heap structure로 구성되기 때문에 heappop()을 하면 순식간에 min값이 출력된다.
# 그래도 heap_st_li 의 첫 번째 index는 항상 최소값을 보장한다.

# reference: rejam
if __name__ == '__main__':
    T = int(input())
    heap_st_li = [] # heap-structured list
    
    for _ in range(0, T):
        input_ = list(map(int, input().split(' ')))
        command = input_[0]
        if len(input_) > 1:
            data = input_[1]
        
        if command == 1:
            heapq.heappush(heap_st_li, data)

        elif command == 2:
            idx = heap_st_li.index(data)
            heap_st_li.remove(data)
            if idx == 0:
                heapq.heapify(heap_st_li)
            
        elif command == 3:
            print(heap_st_li[0]) # 항상 
            #heapq.heappop(heap_st_li)
