CARTPOLE_V0_EPISODES = 100
CARTPOLE_V0_SUCCESS_AVERAGE = 195


def check_cartpole_v0(trajectory_rewards):
    assert len(trajectory_rewards) == CARTPOLE_V0_EPISODES
    avg_score = (
        sum([sum(rewards) for rewards in trajectory_rewards])
        / CARTPOLE_V0_EPISODES
    )

    print(f"Your average score is {avg_score}.")

    return avg_score >= CARTPOLE_V0_SUCCESS_AVERAGE
