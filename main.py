import pygame
import random
import neat
import os
import pickle

pygame.init()

# Display
WIDTH = 800
HEIGHT = 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Catch the Fruits")

FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Load Images
player_img = pygame.image.load('Assets/player.png').convert_alpha()
player_img = pygame.transform.scale(player_img, (100, 100))

bg = pygame.image.load('Assets/bg.png')
bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))

fruits = [
    pygame.image.load('Assets/fruit0.png').convert_alpha(),
    pygame.image.load('Assets/fruit1.png').convert_alpha(),
    pygame.image.load('Assets/fruit2.png').convert_alpha(),
    pygame.image.load('Assets/fruit3.png').convert_alpha(),
    pygame.image.load('Assets/fruit4.png').convert_alpha()
]

fruits_img = [pygame.transform.scale(img, (30, 30)) for img in fruits]

bomb_img = pygame.image.load('Assets/bomb.png').convert_alpha()
bomb_img = pygame.transform.scale(bomb_img, (30, 30))

# Fonts
small_font = pygame.font.SysFont(None, 30)

class Player:
    SPEED = 10

    def __init__(self, x, y, width, height, image):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.image = image

    def draw(self, win):
        win.blit(self.image, (self.x, self.y))

    def move(self, right=True):
        if right:
            self.x += self.SPEED
        else:
            self.x -= self.SPEED
        self.x = max(0, min(self.x, WIDTH - self.width))  # Ensure player stays within screen boundaries

class Fruit:
    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image
        self.width = image.get_width()
        self.height = image.get_height()
        self.y_vel = random.randint(4, 10)

    def draw(self, win):
        win.blit(self.image, (self.x, self.y))

    def move(self):
        self.y += self.y_vel

    def collision_with_player(self, player):
        if (player.x < self.x + self.width and player.x + player.width > self.x and
            player.y < self.y + self.height and player.y + self.height > self.y):
            return True
        return False

class Bomb(Fruit):
    def __init__(self, x, y, image):
        super().__init__(x, y, image)

def draw_game(win, player, items, score):
    win.blit(bg, (0, 0))
    player.draw(win)
    for item in items:
        item.draw(win)
    score_text = small_font.render("Score: " + str(score), True, BLACK)
    win.blit(score_text, (WIDTH - score_text.get_width() - 10, 10))
    pygame.display.update()

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    player = Player((WIDTH - 100) // 2, HEIGHT - 100, 100, 100, player_img)
    items = []
    fruit_spawn_timer = 0
    fruit_spawn_rate = 500
    score = 0
    run = True
    clock = pygame.time.Clock()
    genome.fitness = 0

    while run:
        dt = clock.tick(FPS)
        fruit_spawn_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if fruit_spawn_timer >= fruit_spawn_rate:
            if random.random() < 0.8:
                new_item = Fruit(random.randint(0, WIDTH - 30), 0, random.choice(fruits_img))
            else:
                new_item = Bomb(player.x, 0, bomb_img)
            items.append(new_item)
            fruit_spawn_timer = 0

        player_x_norm = player.x / WIDTH
        nearest_fruit = None
        nearest_bomb = None
        min_fruit_dist = float('inf')
        min_bomb_dist = float('inf')

        for item in items:
            if isinstance(item, Fruit):
                dist = ((item.x - player.x) ** 2 + (item.y - player.y) ** 2) ** 0.5
                if dist < min_fruit_dist:
                    min_fruit_dist = dist
                    nearest_fruit = item
            elif isinstance(item, Bomb):
                dist = ((item.x - player.x) ** 2 + (item.y - player.y) ** 2) ** 0.5
                if dist < min_bomb_dist:
                    min_bomb_dist = dist
                    nearest_bomb = item

        nearest_fruit_x_norm = nearest_fruit.x / WIDTH if nearest_fruit else 0
        nearest_fruit_y_norm = nearest_fruit.y / HEIGHT if nearest_fruit else 0
        nearest_fruit_dist_norm = min_fruit_dist / ((WIDTH ** 2 + HEIGHT ** 2) ** 0.5) if nearest_fruit else 0
        nearest_fruit_y_vel = nearest_fruit.y_vel / 10 if nearest_fruit else 0

        nearest_bomb_x_norm = nearest_bomb.x / WIDTH if nearest_bomb else 0
        nearest_bomb_y_norm = nearest_bomb.y / HEIGHT if nearest_bomb else 0
        nearest_bomb_dist_norm = min_bomb_dist / ((WIDTH ** 2 + HEIGHT ** 2) ** 0.5) if nearest_bomb else 0
        nearest_bomb_y_vel = nearest_bomb.y_vel / 10 if nearest_bomb else 0

        output = net.activate((
            player_x_norm,
            nearest_fruit_x_norm,
            nearest_fruit_y_norm,
            nearest_fruit_dist_norm,
            nearest_fruit_y_vel,
            nearest_bomb_x_norm,
            nearest_bomb_y_norm,
            nearest_bomb_dist_norm,
            nearest_bomb_y_vel
        ))

        move_decision = output[0]
        action_decision = output[1]

        if action_decision > 0.5:
            if move_decision < 0.5:
                player.move(right=False)
            else:
                player.move(right=True)

        genome.fitness += 0.1

        for item in items[:]:
            item.move()
            if item.y > HEIGHT:
                items.remove(item)
            elif isinstance(item, Bomb) and item.collision_with_player(player):
                genome.fitness -= 5
                run = False
            elif isinstance(item, Fruit) and item.collision_with_player(player):
                items.remove(item)
                score += 1
                genome.fitness += 50  # Increased reward for collecting fruits

        draw_game(WINDOW, player, items, score)

def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Add the checkpoint reporter to save checkpoints every 10 generations
    p.add_reporter(neat.Checkpointer(10))

    winner = p.run(eval_genomes, 50)

    print(f'\nBest genome:\n{winner}')

    # Save the winner genome
    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        eval_genome(genome, config)

def load_and_play_winner(config_file, best_genome_path):
    # Load config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Load the best genome
    with open(best_genome_path, 'rb') as f:
        winner = pickle.load(f)

    # Run the genome
    eval_genome(winner, config)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    # Check if a checkpoint exists
    checkpoint_file = 'neat-checkpoint-0'
    checkpoint_path = os.path.join(local_dir, checkpoint_file)

    print(f"Current working directory: {os.getcwd()}")
    print(f"Checkpoint path: {checkpoint_path}")

    if os.path.isfile(checkpoint_path):
        print(f"Checkpoint file {checkpoint_file} found, restoring checkpoint.")
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    else:
        print(f"Checkpoint file {checkpoint_file} not found, starting new population.")
        p = neat.Population(neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                               neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                               config_path))

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Add the checkpoint reporter to save checkpoints every 10 generations
    p.add_reporter(neat.Checkpointer(10))

    winner = p.run(eval_genomes, 50)

    print(f'\nBest genome:\n{winner}')

    # Save the winner genome
    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)
