# Play an episode according to a given policy
# env: environment
# policy: function(state)
# render: whether to render the episode or not (default - False)
def play_episode(env, policy, render=False):
    states, actions, rewards = [], [], []
    
    obs, _ = env.reset()  # ✅ Unpack (obs, info)
    states.append(obs)
    
    done = False
    if render:
        env.render()

    while not done:
        action = policy(states[-1])
        actions.append(action)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if render:
            env.render()

        states.append(obs)
        rewards.append(reward)

    return states, actions, rewards

# Play an episode according to a given policy and add to a replay buffer
# env: environment
# policy: function(state)
# buf: replay buffer
# returns states, actions, rewards, and state-action sequence
def play_episode_rb(env, policy, buf):
    states, actions, rewards, state_action_sequence = [], [], [], []

    obs, _ = env.reset()  # ✅ Unpack (obs, info)
    states.append(obs)
    state_action_sequence.append(obs)

    done = False
    while not done:
        action = policy(states[-1])
        actions.append(action)
        state_action_sequence.append(action)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        state_action_sequence.append(next_obs)

        done = terminated or truncated
        buf.add(states[-1], action, reward, next_obs, done)

        states.append(next_obs)
        rewards.append(reward)

    return states, actions, rewards, state_action_sequence
