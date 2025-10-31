from collections import deque # for snake body representation
from enum import Enum # for direction enum
from typing import List, Tuple # for type hints
import heapq # for priority queue
import random # for random food spawning
import time # for sleep
import os # for clearing console

# direction enum for snake movement
class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

# grid class to manage the game board
class Grid:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.foodCoordinates = (-1, -1)
        '''create grid'''
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]

    def spawnFood(self, snake: "Snake"):
        random.seed()
        if self.foodCoordinates != (-1, -1):
            return
        snake_set = set(snake.snakeBody)
        while True:
            x = random.randint(1, self.rows - 2)
            y = random.randint(1, self.cols - 2)
            if (x, y) not in snake_set:
                self.grid[x][y] = 'F'
                self.foodCoordinates = (x, y)
                break

    def render_lines(self, snake: "Snake", explored=None, path=None) -> List[str]:
        top_border = '+' + '-' * self.cols + '+'
        lines: List[str] = [top_border]
        snake_set = set(snake.snakeBody)
        head = snake.snakeBody[0]
        explored_set = set(explored or [])
        path_set = set(path or [])
        if head in path_set:
            path_set.discard(head)
        for row in range(self.rows):
            row_chars = []
            for col in range(self.cols):
                coord = (row, col)
                if coord == head:
                    row_chars.append('@')
                elif coord in snake_set:
                    row_chars.append('o')
                elif self.grid[row][col] == 'F':
                    row_chars.append('F')
                elif coord in path_set:
                    row_chars.append('*')
                elif coord in explored_set:
                    row_chars.append('.')
                else:
                    row_chars.append(' ')
            lines.append('|' + ''.join(row_chars) + '|')
        lines.append(top_border)
        return lines

    '''draw the rendered lines'''
    def drawGrid(self, snake: "Snake"):
        os.system('cls' if os.name == 'nt' else 'clear')
        for line in self.render_lines(snake):
            print(line)

    def isValid(self, snake: "Snake", heading: Direction) -> bool:
        """return false if moving in dir that would hit wall/snake."""
        head_r, head_c = snake.snakeBody[0]
        
        if heading == Direction.UP:
            if head_r == 0: return False
        if heading == Direction.DOWN:
            if head_r == self.rows - 1: return False
        if heading == Direction.LEFT:
            if head_c == 0: return False
        if heading == Direction.RIGHT:
            if head_c == self.cols - 1: return False
        return True  

    def dijkstra(self, snake: "Snake"):
        target = self.foodCoordinates
        start = snake.snakeBody[0]
        infinity = float('inf')

        distance = [[infinity for _ in range(self.cols)] for _ in range(self.rows)]
        explored = set()
        explored_order = []
        priority_queue = []

        '''dijkstra logic here'''
        sr, sc = start
        distance[sr][sc] = 0
        # 2d list -> store the coordinates of the cells predecessor
        previous = [[None for _ in range(self.cols)] for _ in range(self.rows)] 
        heapq.heappush(priority_queue, (0, sr, sc))  # 0: the priority (distance from start)

        while priority_queue:
            dist, r, c = heapq.heappop(priority_queue)  # removes and returns at the same time
            current = (r, c)

            if current in explored:
                continue
            explored.add(current)
            explored_order.append(current)

            if current == target:
                break
            # 4-neighborhood (up, down, left, right)
            for nr in range(max(0, r - 1), min(self.rows, r + 2)):
                for nc in range(max(0, c - 1), min(self.cols, c + 2)):
                    # skip diagonals and self
                    if abs(nr - r) + abs(nc - c) != 1:
                        continue
                    # skip snake body
                    if (nr, nc) in snake.snakeBody:
                        continue
                    # push neighbours to pq
                    new_dist = dist + 1
                    if new_dist < distance[nr][nc]:
                        distance[nr][nc] = new_dist
                        heapq.heappush(priority_queue, (new_dist, nr, nc))
                        previous[nr][nc] = (r, c)
        return distance, previous, explored_order
    
    def a_star(self, snake: "Snake"):
        start = snake.snakeBody[0]
        sr, sc = start
        target = self.foodCoordinates
        target_r, target_c = target
        infinity = float('inf')
        
        '''heuristic based on Manhattan distance'''
        h_score = [
            [abs(target_r - row) + abs(target_c - col) for col in range(self.cols)]
            for row in range(self.rows)]
        
        '''g-score: actual distance'''
        # Initialize G-score to infinity for every cell
        g_score = [[infinity for _ in range(self.cols)] for _ in range(self.rows)]
        g_score[sr][sc] = 0 # mark start as 0
        explored = set()
        explored_order = []
        priority_queue = []
        previous = [[None for col in range(self.cols)] for row in range(self.rows)]
        start_f_score = h_score[sr][sc]
        heapq.heappush(priority_queue, (start_f_score, sr, sc))

        while priority_queue:
            f, r, c = heapq.heappop(priority_queue)
            current = r,c


            if current in explored:
                continue
            explored.add(current)
            explored_order.append(current)
            
            if (r, c) == target:
                break

            # 4-neighborhood (up, down, left, right)
            for nr in range(max(0, r - 1), min(self.rows, r + 2)):
                for nc in range(max(0, c - 1), min(self.cols, c + 2)):
                    # skip diagonals and self
                    if abs(nr - r) + abs(nc - c) != 1:
                        continue
                    # skip snake body
                    if (nr, nc) in snake.snakeBody:
                        continue
                    
                    # push neighbours to pq if distance got better
                    '''f_score = g_score + h_score'''
                    '''g_score = cost so far'''
                    new_dist = g_score[r][c]+ 1
                    '''if better path was found'''
                    if new_dist < g_score[nr][nc]:
                        g_score[nr][nc] = new_dist
                        '''Total Cost): F= G + H This is the priority used in the queue'''
                        f_score = new_dist + h_score[nr][nc] # f_score is what determines which node gets expanded next 
                        heapq.heappush(priority_queue, (f_score, nr, nc))
                        previous[nr][nc] =  r, c
        return g_score, previous, explored_order

         
'''returns path as a list of coordinates'''
def reconstruct_path(previous, start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
    if target == (-1, -1):
        return []
    path: List[Tuple[int, int]] = []
    node = target
    while node != start:
        path.append(node)
        r, c = node
        node = previous[r][c]
        if node is None:
            return []
    path.append(start)
    path.reverse()
    return path

'''returns path wi/ explored set'''
def astar_pathfinder(grid: Grid, snake: "Snake") -> Tuple[List[Tuple[int, int]], set]:
    _, previous, explored = grid.a_star(snake)
    target = grid.foodCoordinates
    start = snake.snakeBody[0]
    path = reconstruct_path(previous, start, target)
    return path, set(explored)

'''returns path wi/ explored set'''
def dijkstra_pathfinder(grid: Grid, snake: "Snake") -> Tuple[List[Tuple[int, int]], set]:
    _, previous, explored = grid.dijkstra(snake)
    target = grid.foodCoordinates
    start = snake.snakeBody[0]
    path = reconstruct_path(previous, start, target)
    return path, set(explored)

'''maps string to corresponidng algorithm'''
def get_pathfinder(algorithm: str):
    if algorithm == "dijkstra":
        return dijkstra_pathfinder
    if algorithm == "a_star":
        return astar_pathfinder
    print(f"Unknown pathfinding algorithm: {algorithm}")

# snake class to manage the snake body and movement
class Snake:
    def __init__(self, start_pos=(10, 10)):
        self.snakeBody = deque([start_pos])
    
    def snakeDirection(self, r: int, c: int):
        # head coordinates
        hr, hc = self.snakeBody[0]
        if r > hr and c == hc:
            return Direction.DOWN
        if r < hr and c == hc:
            return Direction.UP
        if r == hr and c < hc:
            return Direction.LEFT
        if r == hr and c > hc:
            return Direction.RIGHT
        return None
        
    def moveSnake(self, grid: Grid, algorithm: str):
        """Update snake position; grow if food eaten."""
        pathfinder = get_pathfinder(algorithm)

        while True:
            result = pathfinder(grid, self)
            if isinstance(result, tuple) and len(result) == 2:
                directions, _ = result
            else:
                directions = result
            target = grid.foodCoordinates

            if not directions or len(directions) < 2:
                break
            
            '''move snake until he reaches food'''
            for coordinate in directions[1:]:
                r, c = coordinate
                heading = self.snakeDirection(r, c)
                if heading is None or not grid.isValid(self, heading):
                    break
                self.snakeBody.appendleft(coordinate)
                if coordinate == target:
                    grid.grid[r][c] = 0
                    grid.foodCoordinates = (-1, -1)
                    grid.spawnFood(self)
                else:
                    self.snakeBody.pop()
                grid.drawGrid(self)
                time.sleep(0.01)


class AlgorithmRun:
    def __init__(self, name: str, algorithm: str, rows: int, cols: int):
        self.name = name
        self.algorithm = algorithm
        self.pathfinder = get_pathfinder(algorithm)
        self.grid = Grid(rows, cols)
        self.snake = Snake(start_pos=(rows // 2, cols // 2))
        self.grid.spawnFood(self.snake)
        self.score = 0
        self.steps = 0
        #self.step_limit = step_limit or rows * cols
        self.start_time = None
        self.end_time = None
        self.status = "Ready"
        self.error = None
        self.finished = False
        self.last_path: List[Tuple[int, int]] = []
        self.last_explored = set()

    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    def is_active(self) -> bool:
        return not (self.finished or self.error)

    def _mark_finished(self, status: str):
        self.finished = True
        self.status = status
        if self.start_time is None:
            self.start_time = time.time()
        self.end_time = time.time()

    def step(self):
        if not self.is_active():
            return

        if self.start_time is None:
            self.start_time = time.time()

        if self.grid.foodCoordinates == (-1, -1):
            self.grid.spawnFood(self.snake)

        try:
            result = self.pathfinder(self.grid, self.snake)
        except Exception as exc:
            self.error = str(exc)
            self._mark_finished("Error")
            return

        if isinstance(result, tuple) and len(result) == 2:
            directions, explored = result
        else:
            directions, explored = result, set()

        self.last_path = directions or []
        self.last_explored = set(explored) if explored else set()

        if not directions or len(directions) < 2:
            self._mark_finished("No path to food")
            return

        next_coordinate = directions[1]
        heading = self.snake.snakeDirection(*next_coordinate)
        if heading is None or not self.grid.isValid(self.snake, heading):
            self._mark_finished("Collision predicted")
            return

        self.snake.snakeBody.appendleft(next_coordinate)
        if next_coordinate == self.grid.foodCoordinates:
            r, c = next_coordinate
            self.grid.grid[r][c] = 0
            self.grid.foodCoordinates = (-1, -1)
            self.score += 1
        else:
            self.snake.snakeBody.pop()

        self.steps += 1
        self.status = "Running"

        if self.grid.foodCoordinates == (-1, -1):
            self.grid.spawnFood(self.snake)


    def board_lines_with_meta(self) -> List[str]:
        board_lines = self.grid.render_lines(self.snake, explored=self.last_explored, path=self.last_path)
        title = f"{self.name} ({self.algorithm})"
        score_line = f"Score: {self.score}"
        time_line = f"Time: {self.elapsed_time():.2f}s"
        status_line = f"Status: {self.status}" if not self.error else f"Error: {self.error}"
        steps_line = f"Steps: {self.steps}"
        explored_line = f"Explored: {len(self.last_explored)}"
        path_line = f"Path len: {len(self.last_path)}"
        meta = [title, score_line, time_line, steps_line, explored_line, path_line, status_line, ""]
        width = max(
            max(len(line) for line in board_lines),
            max(len(line) for line in meta)
        )
        meta = [line.ljust(width) for line in meta]
        board = [line.ljust(width) for line in board_lines]
        return meta + board


def render_runs(runs: List[AlgorithmRun]):
    os.system('cls' if os.name == 'nt' else 'clear')
    columns = [run.board_lines_with_meta() for run in runs]
    widths = [max(len(line) for line in col) if col else 0 for col in columns]
    max_lines = max(len(col) for col in columns)
    for idx, col in enumerate(columns):
        width = widths[idx]
        columns[idx] = [line.ljust(width) for line in col] + [' ' * width] * (max_lines - len(col))
    for row in zip(*columns):
        print('   '.join(row))


def main():
    rows, cols = 60, 60
    runs = [
        AlgorithmRun("Dijkstra", "dijkstra", rows, cols),
        AlgorithmRun("A*", "a_star", rows, cols),
    ]

    render_runs(runs)
    while True:
        active = False
        for run in runs:
            if run.is_active():
                run.step()
                if run.is_active():
                    active = True
        render_runs(runs)
        if not active:
            break
        time.sleep(0.02)


if __name__ == "__main__":
    main()
