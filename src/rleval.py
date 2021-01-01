CARTPOLE_V0_EPISODES = 100
CARTPOLE_V0_SUCCESS_AVERAGE = 195

LUNARLANDER_V2_EPISODES = 100
LUNARLANDER_V2_SUCCESS_AVERAGE = 200


def check_cartpole_v0(trajectory_rewards):
    assert len(trajectory_rewards) == CARTPOLE_V0_EPISODES
    avg_score = (
        sum([sum(rewards) for rewards in trajectory_rewards])
        / CARTPOLE_V0_EPISODES
    )

    print(f"Your average score is {avg_score}.")

    return avg_score >= CARTPOLE_V0_SUCCESS_AVERAGE


def check_lunarlander_v2(trajectory_rewards):
    assert len(trajectory_rewards) == LUNARLANDER_V2_EPISODES

    avg_score = (
        sum([sum(rewards) for rewards in trajectory_rewards])
        / LUNARLANDER_V2_EPISODES
    )

    print(f"Your average score is {avg_score}.")

    return avg_score >= LUNARLANDER_V2_SUCCESS_AVERAGE


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