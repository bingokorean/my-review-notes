# Binary Trees

> The method of solution involves the development of a theory of finite automata operating on infinite trees. - "Decidability of Second Order Theories and Automata on Trees," M.O.Rabin,1969

Formally, a binary tree는 empty 이거나 a left binary tree와 a right binary tree를 가지는 a root node r 이다. Subtree 자체가 binary tree이다. Left binary tree를 left subtree of the root라 부르고, right binary tree는 right subtree of the root라 부를 수 있다. 

Binary tree는 대부분, 정렬된 key가 저장되는 binary search tree에서 사용된다. 뿐만 아니라 binary tree를 사용하는 많은 application이 있다. High-level 관점에서 보면, binary tree는 hierarchy를 다룰 때 적용하기 적합하다.

다음은 binary tree의 graphical representation이다. Node A는 root이다. Node B와 I는 A의 left와 right children이다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/interview_py/images/binary_trees_1.PNG" width="90%" height="90%"></p>

Node는 additional data를 저장한다. Node의 prototype은 다음과 같다.

```
class BinaryTreeNode:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
```

Root를 제외한 모든 node는 left subtree의 root이거나 right subtree의 root가 될 수 있다. l이 p의 left subtree의 root라고 하면, l은 p의 left child라 할 수 있고, p는 l의 parent라 할 수 있다 (right child도 같은 이야기). 한 node가 p의 left child 또는 right child일 경우, 해당 node를 p의 child라 부른다. 주목할 점은 root를 제외하고, 모든 node는 unique parent를 가진다. 항상 그렇지는 않지만, node object를 정의할 때 parent field를 포함한다 (root node는 null로). 각 node는 root에서부터 자기자신의 node까지의 unique sequence of nodes 정보를 가질 수 있다. 이러한 sequence는 search path라 부르기도 한다. 


