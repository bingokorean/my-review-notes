########################################################################
""" Queue using Two Stacks """
########################################################################
''' by saikiran9194, Medium, 30
In this challenge, you must first implement a queue using two stacks. 
Then process q queries, where each query is one of the following 3 types:
   1 x: Enqueue element x into the end of the queue.
   2: Dequeue the element at the front of the queue.
   3: Print the element at the front of the queue.
### Sample Input
10
1 42
2
1 14
3
1 28
3
1 60
1 78
2
2
### Sample Output
14
14
'''

# 2개의 stack으로 구현해야 하지만... 다음과 같이 약식으로 구현했다.
# 다음에 기회가 있으면 2개의 stack으로 해결해보자 (훨씬 더 어려운 듯)
if __name__ == '__main__':
    T = int(input())
    queue_li = []
    
    for _ in range(0, T):
        input_ = list(map(int, input().split(' ')))
        command = input_[0]
        if len(input_) > 1:
            data = input_[1]
            
        if command == 1:
            queue_li.append(data)

        elif command == 2:
            queue_li.pop(0)
            
        elif command == 3:
            print(queue_li[0])       
            
            
            
            
