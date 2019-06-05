# Binary Trees

> The method of solution involves the development of a theory of finite automata operating on infinite trees. - "Decidability of Second Order Theories and Automata on Trees," M.O.Rabin,1969

Formally, a binary tree는 empty 이거나 a left binary tree와 a right binary tree를 가지는 a root node r 이다. Subtree 자체가 binary tree이다. Left binary tree를 left subtree of the root라 부르고, right binary tree는 right subtree of the root라 부를 수 있다. 

Binary tree는 대부분, 정렬된 key가 저장되는 binary search tree에서 사용된다. 뿐만 아니라 binary tree를 사용하는 많은 application이 있다. High-level 관점에서 보면, binary tree는 hierarchy를 다룰 때 적용하기 적합하다.

다음은 binary tree의 graphical representation이다. Node A는 root이다. Node B와 I는 A의 left와 right children이다.

<p align="center"><img src="https://github.com/gritmind/review/blob/master/code/book/interview_py/images/binary_trees_1.PNG" width="90%" height="90%"></p>

Node는 additional data를 저장한다. Node의 prototype은 다음과 같다.

```python
class BinaryTreeNode:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
```

Root를 제외한 모든 node는 left subtree의 root이거나 right subtree의 root가 될 수 있다. l이 p의 left subtree의 root라고 하면, l은 p의 left child라 할 수 있고, p는 l의 parent라 할 수 있다 (right child도 같은 이야기). 한 node가 p의 left child 또는 right child일 경우, 해당 node를 p의 child라 부른다. 주목할 점은 root를 제외하고, 모든 node는 unique parent를 가진다. 항상 그렇지는 않지만, node object를 정의할 때 parent field를 포함한다 (단, root node는 null로). 각 node는 root에서부터 자기자신의 node까지의 unique sequence of nodes 정보를 가질 수 있다. 이러한 sequence는 search path라 부르기도 한다. 

Parent-child 관계는 binary tree에서 ancestor-descendant 관계로 정의한다. 구체적으로, 어떤 node가 d node의 anscestor라고 한다면, 그 node는 root에서부터 d까지의 search path에 포함된다고 할 수 있고, d node는 그 node의 descendant라 할 수 있다. (저자의 convention으로) Node 자기 자신은 ancestor 그리고 descendant가 될 수 있다. 자기 자신을 제외하고 descendant가 없는 node를 leaf라고 부른다.

Node n의 depth는 root에서부터 n까지의 (n 자기자신은 제외) search path에 포함된 node 개수를 말한다. Binary tree의 height는 tree에 있는 node들 중에서 가장 큰 depth를 말한다. Tree의 level은 모든 node가 똑같은 depth를 가지는 경우에만 정의할 수 있다. (위 그림 참고)

(Figure 9.1 참고) Node I는 J와 O의 parent node이다. Node G는 B의 descendant node이다. Node L까지의 search path는 <A,I,J,K,L>이다. Node N의 depth는 4이다. Node M은 maximum depth 5를 가지고, 이에 따라 tree의 height는 5이다. (주의) Node B를 root(기준)로 하고 이의 subtree의 height는 3이다. 반면, the height of the subtree rooted at H는 0이다. Node D, E, H, M, N, P는 tree의 leaf들이다.

Full binary tree란 leaf node를 제외한 모든 node가 2개의 children을 가지고 있는 tree를 말한다. Perfect binary tree는 full binary tree임과 동시에 모든 leaf node들이 똑같은 depth를 가진 tree를 말한다. Complete binary tree는 (아마도) 마지막을 제외한 모든 레벨이 완벽히 채워져 있고, 모든 노드들이 최대한 왼쪽인 상태이다. Full binary tree의 nonleaf node의 개수는 leaf node의 개수보다 적다. Perfect binary tree의 height h는 정확하게 2^(h+1)-1 nodes를 가지고 2^h는 leaf들이다. n node를 가지는 complete binary tree의 height는 [logn]이다. Left-skewed tree는 right child가 단 한개도 없는 node들로 구성된다. Right-skewed tree는 반대로 left child가 단 한개도 없는 node들로 구성된다. 위의 두 가지 중 하나라면, 우리는 binary tree를 skewed 되었다고 표현한다.

Binary tree의 key computation은 tree의 모든 노드들을 **traversing**하는 것이다. (Traversing 또는 walking이라고도 불린다) 

* [Inorder traversal] traverse the left subtree, visit the root, then traverse the right subtree. <D,C,E,B,F,H,G,Z,J,L,M,K,N,I,O,P>
* [Preorder traversal] visit the root, traverse the left subtree, then traverse the right subtree. <A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P>
* [Postorder traversal] traverse the left subtree, traverse the right subtree, and then visit the root. <D,E,C,H,G,F,B,M,L,N,K,J,P,O,I,A>

n nodes와 height h를 가지는 T라는 binary tree가 있다고 하자. Recursively하게 구현하면, traversal은 O(n) time complexity와 O(h) addtional space complexity를 가진다. (space complexity는 maximum depth of the function call stack이다). 각 노드가 parent field를 가질 때 마다 traversal은 추가적인 O(1) space complexity를 가진다. (tree 용어는 overloaded되었으므로 혼란을 불러일으킬 수 있다)

### Top Tips for Binary Trees

* Recursive 알고리즘은 tree 문제에 잘 어울린다. Function call stack을 할당할 때마다 space를 추가하는 것을 기억하자.
* 어떤 tree 문제는 O(n)을 가지는 간단한 brute-force 솔루션을 가지지만, existing tree node를 사용하여 O(1)까지 줄일 수 있는 영리한 솔루션도 가진다.
* Complexity 분석을 할 때, left- and right-skewed tree를 고려하자. h height인 tree의 O(h) complexity를 balanced tree로 만들면 O(logn)으로 변환할 수 있다. 하지만, skewed tree는 O(n) complexity를 가진다.
* 각 node가 parent field를 가지면, code를 더 심플하게 만들 수 있고, time과 space complexity를 줄일 수 있다.
* 실수로 single child를 leaf로 판단할 수 있으니 조심하자.

### Binary trees boot camp

binary tree에 speed를 높이기 위한 좋은 방법은 다음과 같은 세 가지의 기본적인 traversal (inorder, preorder, postorder)을 구현하는 것이다.

```python
def tree_traversal(root):
    if root: # 이게 recursive 함수의 탈출 조건이 되기도 한다.
	    # Preorder: Processes the root before the traversals of left and right
		# children.
		print('Preorder: %d' % root.data)
		tree_traversal(root.left)
		# Inorder: Processes the root after the traversal of left child and
		# before the traversal of right child
		print('Inorder: %d' % root.data)
		tree_traversal(root.right)
		# Postorder: Processes the root after the traversals of left and right
		# children
		print('Postorder: %d' % root.data)
```

각 approach의 time complexity는 O(n)이다 (n는 tree에서 node의 개수임). Function call stack은 최대 tree의 depth h를 가지므로 space complexity는 O(h)이다. Height h의 최소값은 logn이고 (complete binary tree), h의 최대값은 n이다 (skewed tree).












