import gym
import numpy as np
from pprint import pprint

class PolicyIterationAgent:
    '''
    Класс, эмулирующий работу агента
    '''
    def __init__(self, env):
        self.env = env
        # Пространство состояний
        self.observation_dim = env.observation_space.n
        # Пространство действий
        self.action_dim = env.action_space.n
        # Задание стратегии (политики)
        self.policy_probs = np.full((self.observation_dim, self.action_dim), 1 / self.action_dim)
        # Начальные значения для v(s)
        self.state_values = np.zeros(shape=(self.observation_dim))
        # Начальные значения параметров
        self.maxNumberOfIterations = 1000
        self.theta = 1e-6
        self.gamma = 0.99

    def print_policy(self):
        '''
        Вывод матриц стратегии
        '''
        print('Стратегия:')
        pprint(self.policy_probs)

    def policy_evaluation(self):
        '''
        Оценивание стратегии 
        '''
        valueFunctionVector = self.state_values
        for iterations in range(self.maxNumberOfIterations):
            valueFunctionVectorNextIteration = np.zeros(shape=(self.observation_dim))
            for state in range(self.observation_dim):
                action_probabilities = self.policy_probs[state]
                outerSum = 0
                for action, prob in enumerate(action_probabilities):
                    innerSum = 0
                    for probability, next_state, reward, isTerminalState in self.env.P[state][action]:
                        innerSum += probability * (reward + self.gamma * self.state_values[next_state])
                    outerSum += self.policy_probs[state][action] * innerSum
                valueFunctionVectorNextIteration[state] = outerSum
            if np.max(np.abs(valueFunctionVectorNextIteration - valueFunctionVector)) < self.theta:
                valueFunctionVector = valueFunctionVectorNextIteration
                break
            valueFunctionVector = valueFunctionVectorNextIteration
        return valueFunctionVector

    def policy_improvement(self):
        '''
        Улучшение стратегии 
        '''
        qvaluesMatrix = np.zeros((self.observation_dim, self.action_dim))
        improvedPolicy = np.zeros((self.observation_dim, self.action_dim))
        for state in range(self.observation_dim):
            for action in range(self.action_dim):
                for probability, next_state, reward, isTerminalState in self.env.P[state][action]:
                    qvaluesMatrix[state, action] += probability * (reward + self.gamma * self.state_values[next_state])
            bestActionIndex = np.where(qvaluesMatrix[state, :] == np.max(qvaluesMatrix[state, :]))
            improvedPolicy[state, bestActionIndex] = 1 / np.size(bestActionIndex)
        return improvedPolicy

    def policy_iteration(self, cnt):
        '''
        Основная реализация алгоритма
        '''
        for i in range(1, cnt+1):
            self.state_values = self.policy_evaluation()
            self.policy_probs = self.policy_improvement()
        print(f'Алгоритм выполнился за {i} шагов.')

def play_agent(agent):
    env2 = gym.make('Taxi-v3', render_mode='human')
    state = env2.reset()
    if isinstance(state, tuple):
        state = state[0]
    env2.render()
    done = False
    while not done:
        p = agent.policy_probs[state]
        if isinstance(p, np.ndarray):
            action = np.random.choice(agent.action_dim, p=p)
        else:
            action = p
        next_state, reward, terminated, truncated, _ = env2.step(action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        env2.render()
        state = next_state
        if terminated or truncated:
            done = True
    env2.close()

def main():
    # Создание среды
    env = gym.make('Taxi-v3')
    env.reset()
    # Обучение агента
    agent = PolicyIterationAgent(env)
    agent.print_policy()
    agent.policy_iteration(1000)
    agent.print_policy()
    # Проигрывание сцены для обученного агента
    play_agent(agent)

if __name__ == '__main__':
    main()
