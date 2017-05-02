# 5.1 Dynamical Systems

## Contents



## Keyword
* dynamic system
  * difference equation (=map)
  * oridnary differential equation
* attractor (=invariant set)
  * fixed point attractor
  * limit cycle attractor
  * limit torus
  * chaotic attractor
* stretching & folding
* symbolic dynamics
* Edge of chaos (=tangency)
* Structural Stability
  * transient
  * attractor
  * damping terms
  * frictionless
* Dissipative System
  * rhythmic patterns
  * limit-cycle attractors
  * Energy dissipation part
  * Energy Supply part

## Summary
* Dynamical systems' theories lay the groundwork for the synthetic modeling studies to follow.

_[Intuitive Explanation]_
* Dynamical system can be described by the **difference equation** (X_t+1 = G(X_t, P)) (also called a "map") 
  * it can be described at any time. The G function is also called **"map"**.
  * The development of the system state is obtained by iterating the mapping of the current state at t to the next state at t+1 starting from given initial state.
  * The dynamic system is often investigated by examining changes in time-development trajectories versus changes in the representative parameter set P. 
  * If the function G() is as a nonlinear function ???
    * the trajectories of time development can become complex because of the nonlinearity
    * the time development of the state cannot be obtained analytically
    * It can be obtained only through numerical computation as integration over time from a given initial state X_0
    * The computation can only be executed with the use of modern digital computers
* Dynamical system can be described also by **an oridnary differential equation** in continuous time (X^' = F(X, P))
  * X^' is a vector of the time derivative of the state
  * F() is a nonlinear dynmaic function parameterized by P
  * The exact trajectory in continuous time can be obtained also by integrating the time derivative from a given dynamical state at the initial time
  
_[Attractor]_
* The structure of a particular dynamical system is characterized by the configuration of attractors in the system.
* Attractors determine the time evolution profiles of different states. 
* Attractors are basins (=regions) toward which trajectories of dynamical states converge.
* An attractor is called an **invariant set** because after trajectories converge for infite time, they become invariant trajectories.
* When coverged, trajectories are no longer variable and are instead determined, representing **stable state** behaviors characterizing the system.
* Ourside of attractors or invariant sets are transient states wherein trajectories are variable.
* Attractors can be roughly categorized in four types
  1. A fixed point attractor, where all dynamic states converge to a point
  2. A limit cycle attractor, where the trajectory converges to a cyclic oscillation pattern with constant periodicity
  3. A limit torus, where there is more than one frequency involved in the periodic trajectory of the system and two of these frequencies form an irrational fraction. In detail, the trajectory is no longer closed and it exhibits quasi-peirodicity.
  4. A chaotic attractor (=strange attractor), where the trajectory exhibits infinite periodicity and thereby forms fractal structures.
  5. A multiple-local attractor, where multiple local attractors (ex. fixed point or limit cycle attractors) can coexist in the same state space. The system converges depends on the initial state.

_[Discrete Time System (Logistic Map)]_
* Even logistic map is with one-dimensional dynamic state, its behavior is nontrivial. It is written in discrete-time form: x_t+1 = ax_t(1-x_t)
* Let's see how the dynamical structure of a logistic map changes when the parameter a is varied continuously.
  * Less than a=3.0, the trajectory of x_t converges toward a fixed point. The trajectory becomes the fixed-point attractor. 
  * From a=3.0 to a=3.6, the fixed-point attractor bifurcates into a limit-cycle attractor with a period of 2. 
  * More than a=3.6, the limit-cycle attractor becomes a chaotic attractor
* About the chaotic attractor in logistic map
  * no periodicity
  * sensitivity with respect to initial conditions, which generates nonrepeatable behaviors (explained by **stretching & folding**)
  * Even though two trejectories are generated from very similar/closed initial states, two trejectories increase exponentially as iterations progress
  * The iterated stretching and folding is considered to be a general mechanism for generating chaos

_[Symbolic dynamics: Chaotic dynamics <-> Symbolic process]_
* If we observe the output sequence of the logistic map and label it with two symbols, "H" for values greater than 0.5 and "L" for those less than or equal to 0.5, we get probabilistic sequences of alternating "H" and "L". 
* When logistic map generates "H" or "L" with equal probability with no memory (when a=4), like a coin flip, this can be represented by a one-state probabilistic finite state machine (FSM) with an equal probability output for "H" and "L" from this single state. (when changing a value, the number of discrete states and different probability is reconstructred for each)
* It is a symbolic dynamics, which provides a theorem to connect real number dynamical systems and discrete symbol systems.
* "Edge of chaos"
  * the complexity (the number of states) of symbolic dynamics can be from finite to infinite
  * for example, Tangency in nonlinear mapping
  * this generates the phenomena known as intermittent chaos
 
_[Structural Stability]_
* One of properties of nonlinear dynamical systems is the appearance of a particular attractor configuration for any given dynamical system.
* A particular equation for a dynamic system can indicate the direction fo change of state at each local point in terms of a vector field. However, the vector field itself can't tell us what the attractor looks like.
* The attractor emerges only after several number of iterations, through the **transient** process of converging
* [중요]: Attractors as trajectories of steady states can't exist by themselves in isolation. Transident parts of the vector flow make attractors stable.  (-> structural stability of attractors)

_[Example of not-structually-stable]_
* Lt's assume there are systems that generate oscillation patterns (= sinusoidal function, harmonic oscillator)
* Those equations represent a second order dynamic system **without damping terms**
* A frictionless spring-mass system can indeed generate sinusoidal oscillation patterns
* However, such patterns are not structurally stable
  * if we apply force to the mass of the oscillator instantaneously, the amplitude of oscillation will change immediately and the original oscillation pattern will never be recovered automatically (=it is **frictionless**(=외부마찰이없는)
  * if we plot the vector field in (x,v) space, we will see that the vector flow describes concentric circles where there is no convergent flow that constitute a limit-cycle attractor.

_[Dissipative System]_
* Most rhythmic patterns in biological systems are thought to be generated by **limit-cycle attractors** because of their potential stability against perturbations(=노이즈?). -> 리드믹한 패턴의 의미가 주기성을 가진다는 의미와 같다.
* rhythmic patterns generated by central pattern generators in neural circuits for the heart beat, locomotion, breathing, swimming, ...
* In real physical world, limit-cycle attractors are generated by nonlinear dynamical systems called **dissipative systems**
* Dissipative system = 'Energy dissipation part' + 'Energy Supply part'
  * If two parts are balanced, the limit-cycle attractor can be generated. 
  * Energy can be dissipated by dampening caused by friction(=외부마찰 ex.전기선에서 저항)
  * When some amount of energy is supplied due to a perturbation(ex.돌부리에넘어짐) from an external source, the state trajectory deviates and becomes transient(=attractor가되기전잠시머무는곳) 하지만, 이내 곧 들어온 에너지만큼 dissipating를 통해 balance를 유지하면서 original attractor를 되찾는다
* [요약] The structural stability of dynamic patterns in terms of physical movements or neural activity in biological systems can be achieved through attractor dynamics by means of a dissipative structure.
* the particular attractors appearing in different cases are the products of **emergent properties** of such nonlinear (dissipative) dynamic systems.

_[Non-dissipative System]_
* It is a harmonic oscillator without a dampening term. It is not a dissipative system but an energy conservation system.
* Once perturbed, its state trajectory will not return to the original one.









## Summary with My View



## Reference
> This chapter is a part of 'Exploring Robotic Minds' written by Jun Tani. I wrote this summary while taking his class, 'EE817-Deep Learning and Dynamic Neural Network Models'. 
