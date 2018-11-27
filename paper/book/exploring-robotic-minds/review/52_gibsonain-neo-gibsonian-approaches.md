# 5.2 Gibsonian and Neo-Gibsonian Approaches

## Keyword
1. Gibsonian Approach
   * affordance constancy
   * fixed point attractor dynamics
2. Neo-Gibsonian Approaches
   * dissipative structures (entrainment + phase transitions)
   * limit cycle attractor dynamics
3. Infant Developmental Psychology
4. Imitation

| Nitty-Gritty  | Description           |
| ------------- |---------------|
| Affordance      | all possibilities for actions latent in the enviornment |



## Summary
### 1. Gibsonian Approach
A concept called **affordance** has significantly influenced not only psychology, philosophy of the mind, but also synthetic modeling studies such as aritifical intelligence and robotics. **Affordance** is defined by:
  * "all possibilities for actions latent in the enviornment"
  * "behavior relations" that animals are able to acquire in interaction with their environment

Relationships between actors and objects within these environments afford these agents opportunities to generate adequate behaviors.
  * ex. A chair affords sitting on it.

The essential information about the environment comes by way of human processing of the **optical flow** (=motion pattern sensed by eye). It can be used for:
  * to perceive one's own motion pattern
  * to control one's own behavior

With the **optical flow**, Gibson came up with the notion of **affordance constancy**. For example:
  * ex1. a pilot flying toward a target on the ground
    * adjusting the direction of flight so that the focas of expansion (FOE) in the visual optical flow becomes superimposed on the target in order to develop better landing skills (공군훈련방식에 적용)
  * ex2. walking along a corridor while balancing optical flow vectors against both side walls (벽과부딪히지않는 최적의경로)

Each behavior has a crucial **perceptual variable**:
  * the distance between the FOE and target
  * the vector difference between the optical flows for both walls

`Affordance constancy means that body movements are generated to keep these perceptual variables at constant values.` Therefore, if there is coupled dynamics between the environment and a controller in brain, the role of the controller will preserve perceptual constancy. A simple dynamic system theory can show how this constancy may be maintained by assuming the existence of a **fixed point attractor**, which ensures that perceptual variables always converge to a constant state.

[또 다른 예시] Andy Clark Example: how an outfielder positions to catch a fly ball (Clark 1999). Originally, catching action requires complicated calculations of some variables such as the arc, speed, acceleration, and distance of the ball. However, when we use **affordance constancy**, there is a simple strategy to catch it:
  * If the outfielder continues to adjust his movement so that the ball appears in a straight line in his visual field
  * By maintaining this coordination between the inner and the outer for perceptual constancy
  * The **coordination dynamics** appears under simple principles such as perceptual constancy ~~intead of complicated computation~~
  * In other words, the coordination dynamics will converge to a **fixed point attractor**



### 2. Neo-Gibsonian Approaches
In the 1980s, Neo-Gibsonian psychologists started investigating `how to achieve the coordination of many degrees of freedom` by applying the ideas of **dissipative structures** from nonlinear dynamics to psychological observations of human/animal behavior (Scott Kelso, 1995). They considered that the ideas of dissipative structures concerning **limit cycle attractor dynamics**, can serve as a basic principle in organazing **coherent rhythmic movement patterns** such as walking, running, breathing, and hand waving.

The importances of dissipative structure are **entrainment** and **phase transitions**.
1. **Entrainment**: Coupled oscillators initialzed with different phases can converge to a global synchrony with reduced dimensionality by mutual entrainment under certain conditions.
2. **Phase transitions**: the global synchrony can be drastically changed by a shift of an order parameter of the dynamic system by means of phase transition.

[Dissipative structure 실험 - Bimanual finger movement]: (Schoner & Kelso, 1988) 두 개의 손가락을 메트로놈 박자에 맞춰 역위상으로 상하로 antiphase하게 움직인다. 메트로놈 박자속도가 빨라지면, inphase가 되면서 phase trainsition이 일어난다. 즉, 처음엔 antiphase에서 energy가 stable했지만, 나중엔 inphase일때 energy가 stale했다.  Kelso and colleagues showed by computer simulation that the observed dynamic shift is because of the **phase transition** from a particular dynamic structure self-organizing to another, given changes in an order parameter(=메트로놈 스피드) of the system.

Examples including shuch dynamic shift
  * from trot (빨리걷기) to gallop (전력달리기) in horse locomotion
  * from walk to run in human locomotion
  * 일상생활에서도 phase transition을 경험할 수 있다. 예를 들어, walk-run의 중간지점에 있는 경우 그 자세를 유지하는게 에너지 소모가 많다. 걷든지 아니면 뛰든지 둘 중에 하나를 결정해야 에너지가 stable될 수 있다.

[결론] This result showed that behaviors are organized ~~not by an explict central commander top-down~~, but `by implicit synergy among local elements including neurons, muscles, and skeletal mechanics` and showed these behaviors represents emergent characteristics of dissipative structures.


### 3. Infant Developmental Psychology



### 4. Imitation








## Summary with My View



## Reference
> This chapter is a part of 'Exploring Robotic Minds' written by Jun Tani. I wrote this summary while taking his class, 'EE817-Deep Learning and Dynamic Neural Network Models'.
