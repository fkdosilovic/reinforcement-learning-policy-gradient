from sklearn.preprocessing import PolynomialFeatures

identity = PolynomialFeatures(degree=1).fit_transform


def simulate(
    env,
    agent,
    transform=identity,
    render=False,
):
    traj_states = []
    traj_actions = []
    traj_probs = []
    traj_rewards = []

    prev_state = transform(env.reset().reshape(1, -1))
    while True:
        # If we are showing the animation, then render the action.
        if render:
            env.render()
            action = agent.choose_action(state=prev_state)
        else:
            action, probs = agent.sample_action(state=prev_state)

        next_state, reward, is_terminal, _ = env.step(action)

        if not render:
            traj_states.append(prev_state)
            traj_actions.append(action)
            traj_probs.append(probs)
            traj_rewards.append(reward)

        if is_terminal:
            break

        prev_state = transform(next_state.reshape(1, -1))

    return (traj_states, traj_probs, traj_actions, traj_rewards)


def evaluate(env, agent, episodes, check_score, transform=identity):

    trajectory_rewards = []
    for _ in range(episodes):
        rewards = []

        prev_state = transform(env.reset().reshape(1, -1))
        while True:
            action = agent.choose_action(state=prev_state)
            next_state, reward, is_terminal, _ = env.step(action)
            rewards.append(reward)

            if is_terminal:
                break

            prev_state = transform(next_state.reshape(1, -1))

        trajectory_rewards.append(rewards)

    return check_score(trajectory_rewards)
