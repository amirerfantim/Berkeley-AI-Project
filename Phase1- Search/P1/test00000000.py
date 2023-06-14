import util


def bbfs(problem):
    start_state = problem.getStartState()
    goal_state = problem.goal
    forward_queue = util.Queue()
    reverse_queue = util.Queue()
    forward_queue.push((start_state, [], 0))
    reverse_queue.push((goal_state, [], 0))
    forward_visited = set()
    reverse_visited = set()

    while not forward_queue.isEmpty() and not reverse_queue.isEmpty():
        cur_state, actions, costs = forward_queue.pop()
        if cur_state not in forward_visited:
            forward_visited.add(cur_state)
            if cur_state in reverse_visited:
                reverse_actions = reverse_visited[cur_state]
                reverse_actions.reverse()
                return actions + reverse_actions
            for next_state, action, step_cost in problem.getSuccessors(cur_state):
                if next_state not in forward_visited:
                    next_actions = actions + [action]
                    next_cost = costs + step_cost
                    forward_queue.push((next_state, next_actions, next_cost))

        cur_state, actions, costs = reverse_queue.pop()
        if cur_state not in reverse_visited:
            reverse_visited[cur_state] = actions
            if cur_state in forward_visited:
                forward_actions = forward_visited[cur_state]
                forward_actions.reverse()
                return forward_actions + actions

            for next_state, action, step_cost in problem.getSuccessors(cur_state):
                if next_state not in reverse_visited:
                    next_actions = actions + [action]
                    next_cost = costs + step_cost
                    reverse_queue.push((next_state, next_actions, next_cost))

    return []