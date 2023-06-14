# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
        Runs the Value Iteration algorithm to compute the optimal value function.

        Returns:
        None
        """
        for _ in range(self.iterations):
            updated_values = {}

            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    max_value = float('-inf')
                    for action in self.mdp.getPossibleActions(state):
                        q_value = self.computeQValueFromValues(state, action)
                        max_value = max(max_value, q_value)
                    updated_values[state] = max_value
                else:
                    updated_values[state] = 0

            self.values = updated_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q_val = 0
        for t in self.mdp.getTransitionStatesAndProbs(state, action):
            successor_state, probability = t
            reward = self.mdp.getReward(state, action, successor_state)
            q_val += probability * (reward + self.discount * self.values[successor_state])
        return q_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        best_action = None
        best_q_value = float('-inf')

        for action in self.mdp.getPossibleActions(state):
            q_value = self.computeQValueFromValues(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        num_states = len(states)

        for k in range(self.iterations):
            curr_state = states[k % num_states]
            if self.mdp.isTerminal(curr_state):
                continue

            max_q = float('-inf')
            for action in self.mdp.getPossibleActions(curr_state):
                curr_q = self.computeQValueFromValues(curr_state, action)
                if curr_q > max_q:
                    max_q = curr_q
            self.values[curr_state] = max_q
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = {state: set() for state in self.mdp.getStates()}
        priorityQueue = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                max_q = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    if q_value > max_q:
                        max_q = q_value

                    for successor_state, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        if probability > 0:
                            predecessors[successor_state].add(state)

                diff = abs(max_q - self.values[state])
                priorityQueue.push(state, -diff)

        for _ in range(self.iterations):
            if priorityQueue.isEmpty():
                break

            state = priorityQueue.pop()
            if self.mdp.isTerminal(state):
                continue

            actions = self.mdp.getPossibleActions(state)
            max_q = max(self.computeQValueFromValues(state, action) for action in actions)
            self.values[state] = max_q

            for predecessor in predecessors[state]:
                actions = self.mdp.getPossibleActions(predecessor)
                q_values = [self.computeQValueFromValues(predecessor, action) for action in actions]
                difference = abs(self.values[predecessor] - max(q_values))
                if difference > self.theta:
                    priorityQueue.update(predecessor, -difference)

