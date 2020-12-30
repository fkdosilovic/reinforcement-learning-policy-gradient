def simulate(
    env,
    agent,
    render=False,
):
    traj_states = []
    traj_actions = []
    traj_probs = []
    traj_rewards = []

    choose_action = agent.choose_action if render else agent.sample_action

    prev_state = env.reset().reshape(1, -1)
    while True:
        if render:
            env.render()

        action, probs = choose_action(state=prev_state)
        next_state, reward, is_terminal, _ = env.step(action.item())

        traj_states.append(prev_state)
        traj_actions.append(action)
        traj_probs.append(probs)
        traj_rewards.append(reward)

        if is_terminal:
            break

        prev_state = next_state.reshape(1, -1)

    return (traj_states, traj_probs, traj_actions, traj_rewards)


def evaluate(env, agent, episodes, check_score):

    trajectory_rewards = []
    for _ in range(episodes):
        rewards = []

        prev_state = env.reset().reshape(1, -1)
        while True:
            action, _ = agent.choose_action(state=prev_state)
            next_state, reward, is_terminal, _ = env.step(action.item())
            rewards.append(reward)

            if is_terminal:
                break

            prev_state = next_state.reshape(1, -1)

        trajectory_rewards.append(rewards)

    return check_score(trajectory_rewards)
