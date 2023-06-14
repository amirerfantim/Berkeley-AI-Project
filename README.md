# AI Berkeley Pacman Project

The AI Berkeley Pacman Project is a series of assignments and challenges designed to teach students about artificial intelligence concepts and techniques by implementing intelligent agents for the game of Pacman. The project consists of four phases, each building upon the previous one and introducing new challenges and concepts.

- [Phase 1](#Phase 1)
- [Phase 2](#Phase 2)
- [Phase 3](#Phase 3)
- [Phase 4](#Phase 4)


## Phase 1: Search Algorithms

Phase 1 of the AI Berkeley Pacman Project focuses on implementing search algorithms to help Pacman navigate and find paths in a maze. The objective is to implement breadth-first search, depth-first search, uniform cost search, and A* search algorithms.

### Implementation Details

To begin, you will need to understand the basic structure of the project. The Pacman game is represented as a grid, where each cell can be either a wall, food, an empty space, or a ghost. Pacman's task is to find the shortest path to eat all the food pellets while avoiding ghosts.

Your task is to implement the search algorithms by extending the `SearchAgent` class. Each search algorithm has a distinct implementation, but they share common features such as maintaining a fringe (a data structure that holds the unexplored states) and expanding nodes until the goal state is reached.

### Evaluation and Further Challenges

In addition to the basic implementation, there are some optional challenges you can undertake to deepen your understanding of search algorithms. These challenges include implementing heuristics for A* search, experimenting with different cost functions, and dealing with corner cases such as dead ends and loops in the maze.

## Phase 2: Multi-Agent Search

Phase 2 of the AI Berkeley Pacman Project focuses on implementing algorithms for adversarial multi-agent environments. In this phase, Pacman will face not only ghosts but also other Pacman agents, creating a competitive setting. The objective is to implement minimax, alpha-beta pruning, and expectimax algorithms.

### Implementation Details

In this phase, you will extend the `MultiAgentSearchAgent` class and implement the minimax, alpha-beta pruning, and expectimax algorithms. These algorithms will enable Pacman to make optimal decisions by considering the actions of all agents and selecting the best moves accordingly.

### Evaluation and Further Challenges

Once you have implemented the algorithms, you will test them against different game configurations and evaluate their performance based on metrics such as win rate, average game score, and average computation time. You can also experiment with different evaluation functions and search depths to improve Pacman's gameplay.

## Phase 3: Reinforcement Learning

Phase 3 of the AI Berkeley Pacman Project introduces reinforcement learning, a powerful technique for training intelligent agents through interaction with an environment. In this phase, you will implement Q-learning, an off-policy temporal difference learning algorithm, to train Pacman to play optimally.

### Implementation Details

To implement Q-learning, you will extend the `ApproximateQAgent` class and design a feature-based representation of the game state. Features capture relevant information such as Pacman's position, the positions of ghosts, the presence of food, and the score. You will then use these features to update the Q-values, which represent the expected rewards of taking actions in specific states.

### Evaluation and Further Challenges

You will evaluate the performance of your Q-learning agent by testing it against various game scenarios and measuring metrics such as win rate, average game score, and convergence speed. You can further enhance your agent by experimenting with different feature representations, exploring alternative learning algorithms (e.g., SARSA), or applying function approximation techniques to handle large state spaces.

## Phase 4: Ghostbusters

Phase 4 of the AI Berkeley Pacman Project adds a new dimension to the game by introducing sensory perception and probabilistic reasoning. Pacman becomes a ghost
