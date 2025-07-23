import gymnasium as gym
import numpy as np

# 환경, 하이퍼파라미터
env = gym.make('Acrobot-v1')
n_bins = [6,6,6,6]        # 각 θ1, θ2, θ1̇, θ2̇를 6개 구간으로 분할
q_table = np.zeros(n_bins + [env.action_space.n])
alpha, gamma, eps = 0.1, 0.99, 1.0

def discretize(obs):
    # 예시: np.digitize를 써서 continuous→discrete idx 반환
    # 실제로는 각 차원을 min/max로 스케일링해야 함
    return tuple(np.digitize(obs[i], bins=np.linspace(-1,1,n_bins[i]-1))
                 for i in range(4))

for episode in range(10000):
    obs, _ = env.reset()
    state = discretize(obs[:4])  # 첫 4개 값만 사용
    done = False
    while not done:
        # ε-greedy 정책
        if np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize(next_obs[:4])

        # Bellman Q-러닝 업데이트
        best_next = np.max(q_table[next_state])
        q_table[state + (action,)] += alpha * (
            reward + gamma * best_next - q_table[state + (action,)]
        )

        state = next_state

    # ε 점진적 감소
    eps = max(0.01, eps * 0.995)
