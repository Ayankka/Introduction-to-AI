import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# Настройки
GRID_SIZE = 10
MAX_STEPS = 50
EPISODES = 100

# Действия
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# Инициализация
def init_game():
    snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
    direction = random.choice(ACTIONS)
    food = place_food(snake)
    return snake, direction, food

def place_food(snake):
    while True:
        food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if food not in snake:
            return food

def move_snake(snake, direction):
    head_x, head_y = snake[0]
    dx, dy = direction
    return (head_x + dx, head_y + dy)

def is_collision(snake, new_head):
    x, y = new_head
    return not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE) or new_head in snake

def get_state(snake, food):
    head = snake[0]
    fx, fy = food
    hx, hy = head
    return (fx - hx, fy - hy)

def choose_action(state, q_table, epsilon=0.1):
    if random.random() < epsilon or state not in q_table:
        return random.choice(ACTIONS)
    return max(q_table[state], key=q_table[state].get)

def update_q_table(q_table, state, action, reward, next_state, alpha=0.1 , gamma=0.9):
    if state not in q_table:
        q_table[state] = {a: 0 for a in ACTIONS}
    if next_state is not None and next_state not in q_table:
        q_table[next_state] = {a: 0 for a in ACTIONS}
    old_value = q_table[state][action]
    next_max = max(q_table[next_state].values()) if next_state is not None else 0
    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    q_table[state][action] = new_value

# Анимация
def draw(frame, snake_history, food_history):
    plt.clf()
    plt.xlim(-1, GRID_SIZE)
    plt.ylim(-1, GRID_SIZE)
    plt.gca().set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    food = food_history[frame]
    plt.plot(food[0], food[1], 'ro', markersize=12)
    for (x, y) in snake_history[frame]:
        plt.plot(x, y, 'gs', markersize=20)

# Главная часть
if __name__ == "__main__":
    q_table = {}
    scores = []
    best_score = 0
    best_snake_history = []
    best_food_history = []

    for episode in range(EPISODES):
        snake, direction, food = init_game()
        score = 0
        steps = 0
        snake_history = [list(snake)]
        food_history = [food]

        while True:
            state = get_state(snake, food)
            action = choose_action(state, q_table)
            new_head = move_snake(snake, action)

            if is_collision(snake, new_head):
                reward = -10
                update_q_table(q_table, state, action, reward, None)
                break

            snake.insert(0, new_head)

            if new_head == food:
                reward = 10
                score += 1
                food = place_food(snake)
            else:
                reward = -5
                snake.pop()

            next_state = get_state(snake, food)
            update_q_table(q_table, state, action, reward, next_state)

            snake_history.append(list(snake))
            food_history.append(food)

            steps += 1
            if steps >= MAX_STEPS:
                break

        scores.append(score)

        if score > best_score:
            best_score = score
            best_snake_history = snake_history
            best_food_history = food_history

            # 🎥 Визуализация лучшего эпизода прямо во время обучения
            print(f"🎉 Новый лучший результат: {score} (эпизод {episode + 1})")
            plt.ion()
            fig = plt.figure()
            for frame in range(len(best_snake_history)):
                draw(frame, best_snake_history, best_food_history)
                plt.pause(0.1)
            plt.close()

        print(f"Эпизод {episode + 1}: счёт = {score}")



    # 📊 График обучения
    plt.ioff()
    plt.figure()
    plt.plot(scores)
    plt.xlabel('Эпизоды')
    plt.ylabel('Счёт')
    plt.title('Результат обучения змейки')
    plt.grid()
    plt.show()
