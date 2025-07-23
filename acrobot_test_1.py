# acrobot_test.py
import gymnasium as gym

def main():
    # human 모드로 윈도우 창 띄우기
    env = gym.make('Acrobot-v1', render_mode='human')
    obs, info = env.reset(seed=0)

    for step in range(2000):
        env.render()  # 창에 그리기
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"에피소드 종료 → 스텝: {step}, reward: {reward}")
            break

    env.close()

if __name__ == "__main__":
    main()
