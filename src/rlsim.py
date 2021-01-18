def simulate(
    env,
    agent,
    render=False,
    stochastic_policy=True,
):
    traj_states = []
    traj_actions = []
    traj_probs = []
    traj_rewards = []

    choose_action = (
        agent.sample_action if stochastic_policy else agent.choose_action
    )

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

    return (traj_states, traj_actions, traj_probs, traj_rewards)
