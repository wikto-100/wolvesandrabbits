import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

class GrassField:
    """
    A 2D grid of grass levels. Each cell starts at a random density,
    disappears when eaten, and then regrows by a random amount each step.
    """
    def __init__(self, width, height, max_grass, regrowth_rate):
        self.width = width
        self.height = height
        self.max_grass = max_grass
        self.regrowth_rate = regrowth_rate
        # initialize each cell to a random grass level between 0 and max_grass
        self.field = np.random.uniform(0, max_grass, size=(height, width))

    def regrow(self):
        """
        Regrow grass in every cell by a random increment in [0, regrowth_rate],
        capped at max_grass.
        """
        random_growth = np.random.uniform(
            0, self.regrowth_rate, size=(self.height, self.width)
        )
        self.field = np.minimum(self.field + random_growth, self.max_grass)

    def eat(self, x, y, amount):
        """
        Rabbit eats up to `amount` from cell (x,y).
        Returns actual amount eaten; cell level may drop to zero.
        """
        available = self.field[y, x]
        eaten = min(available, amount)
        self.field[y, x] -= eaten
        return eaten


class Rabbit:
    def __init__(self, x, y, health, metabolism):
        self.x = x
        self.y = y
        self.health = health
        self.metabolism = metabolism

    def is_alive(self):
        return self.health > 0

    def move(self, grass_field):
        best = -1.0
        choices = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < grass_field.width and 0 <= ny < grass_field.height:
                    lvl = grass_field.field[ny, nx]
                    if lvl > best:
                        best, choices = lvl, [(nx, ny)]
                    elif lvl == best:
                        choices.append((nx, ny))
        self.x, self.y = random.choice(choices)

    def eat(self, grass_field, eat_amount):
        gained = grass_field.eat(self.x, self.y, eat_amount)
        self.health += gained

    def update(self, grass_field, eat_amount):
        self.move(grass_field)
        self.eat(grass_field, eat_amount)
        self.health -= self.metabolism

    def reproduce(self):
        offspring_h = self.health / 2.0
        self.health = offspring_h
        return [
            Rabbit(self.x, self.y, offspring_h, self.metabolism),
            Rabbit(self.x, self.y, offspring_h, self.metabolism)
        ]


class Wolf:
    def __init__(self, x, y, health, metabolism, eat_amount):
        self.x = x
        self.y = y
        self.health = health
        self.metabolism = metabolism
        self.eat_amount = eat_amount

    def is_alive(self):
        return self.health > 0

    def move(self, rabbits, width, height):
        best = -1
        choices = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    cnt = sum(1 for r in rabbits if r.x == nx and r.y == ny)
                    if cnt > best:
                        best, choices = cnt, [(nx, ny)]
                    elif cnt == best:
                        choices.append((nx, ny))
        self.x, self.y = random.choice(choices)

    def eat(self, rabbits):
        eaten = 0
        survivors = []
        for r in rabbits:
            if eaten < self.eat_amount and r.x == self.x and r.y == self.y:
                eaten += 1
            else:
                survivors.append(r)
        self.health += eaten
        return survivors

    def update(self, sim):
        self.move(sim.rabbits, sim.field.width, sim.field.height)
        sim.rabbits = self.eat(sim.rabbits)
        self.health -= self.metabolism

    def reproduce(self):
        offspring_h = self.health / 2.0
        self.health = offspring_h
        return [
            Wolf(self.x, self.y, offspring_h, self.metabolism, self.eat_amount),
            Wolf(self.x, self.y, offspring_h, self.metabolism, self.eat_amount)
        ]


class Simulation:
    def __init__(
        self, width, height, initial_rabbits, initial_wolves,
        max_grass=4.0, regrowth_rate=0.02,
        rabbit_health=5.0, rabbit_metabolism=0.5, rabbit_eat=2.0, rabbit_thresh=10.0,
        wolf_health=10.0, wolf_metabolism=1.0, wolf_eat=1.0, wolf_thresh=20.0
    ):
        self.field = GrassField(width, height, max_grass, regrowth_rate)
        self.rabbits = [
            Rabbit(random.randrange(width), random.randrange(height),
                   rabbit_health, rabbit_metabolism)
            for _ in range(initial_rabbits)
        ]
        self.wolves = [
            Wolf(random.randrange(width), random.randrange(height),
                 wolf_health, wolf_metabolism, wolf_eat)
            for _ in range(initial_wolves)
        ]
        self.r_params = (rabbit_eat, rabbit_thresh)
        self.w_params = (wolf_eat, wolf_thresh)

    def step(self):
        # Rabbits
        r_eat, r_thresh = self.r_params
        survivors, newborns = [], []
        for r in self.rabbits:
            r.update(self.field, r_eat)
            if not r.is_alive():
                continue
            if r.health > r_thresh:
                newborns.extend(r.reproduce())
            survivors.append(r)
        self.rabbits = survivors + newborns

        # Wolves
        _, w_thresh = self.w_params
        w_surv, w_new = [], []
        for w in self.wolves:
            w.update(self)
            if not w.is_alive():
                continue
            if w.health > w_thresh:
                w_new.extend(w.reproduce())
            w_surv.append(w)
        self.wolves = w_surv + w_new

        # Randomized grass regrowth
        self.field.regrow()


if __name__ == "__main__":
    # PARAMETERS
    W, H           = 50, 50
    INIT_R, INIT_W = 10, 5
    STEPS, INT     = 200, 100

    sim = Simulation(
        W, H, INIT_R, INIT_W,
        max_grass=8.0, regrowth_rate=0.10,
        rabbit_health=10.0, rabbit_metabolism=0.5,
        rabbit_eat=1.0, rabbit_thresh=20.0,
        wolf_health=100.0, wolf_metabolism=0.25,
        wolf_eat=2.0, wolf_thresh=25.0
    )

    fig, (ax_grid, ax_plot) = plt.subplots(1, 2, figsize=(12, 6))
    ax_grid.axis('off')
    img = ax_grid.imshow(np.zeros((H, W, 3)), interpolation='nearest')

    ax_plot.set_xlim(0, STEPS)
    ax_plot.set_ylim(0, max(W*H*sim.field.max_grass, INIT_R+INIT_W)*1.2)
    line_r, = ax_plot.plot([], [], label='Rabbits', color='orange')
    line_g, = ax_plot.plot([], [], label='Total Grass', color='green')
    line_w, = ax_plot.plot([], [], label='Wolves', color='blue')
    ax_plot.legend(loc='upper right')
    ax_plot.set_xlabel('Time Step')
    ax_plot.set_ylabel('Count / Total Grass')

    history_t = []
    history_r = []
    history_g = []
    history_w = []

    def update_frame(frame):
        sim.step()
        if frame == 0:
            history_t.clear()
            history_r.clear()
            history_g.clear()
            history_w.clear()

        # Grid update
        grass_norm = sim.field.field / sim.field.max_grass
        grid = np.zeros((H, W, 3))
        grid[..., 1] = grass_norm
        for r in sim.rabbits:
            grid[r.y, r.x, 0] = 1.0
            grid[r.y, r.x, 1] = 1.0
        for w in sim.wolves:
            grid[w.y, w.x, 2] = 1.0
        img.set_data(grid)

        # Time-series update
        history_t.append(frame)
        history_r.append(len(sim.rabbits))
        history_g.append(sim.field.field.sum())
        history_w.append(len(sim.wolves))

        line_r.set_data(history_t, history_r)
        line_g.set_data(history_t, history_g)
        line_w.set_data(history_t, history_w)

        return img, line_r, line_g, line_w

    ani = animation.FuncAnimation(
        fig, update_frame,
        frames=STEPS, interval=INT, blit=True
    )

    plt.tight_layout()
    plt.show()
