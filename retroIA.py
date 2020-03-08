import gym
import retro

def main():
    for game in retro.data.list_games():
       print(game, retro.data.list_states(game))
    retro.data.list_games()
    env = retro.make(game='SonicTheHedgehog-Genesis')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()