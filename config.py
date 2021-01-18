# Algorithm parameters
H_COST_WEIGHT = 1.7

# Parameters for map generation with Perlin noise
WIDTH = 20
HEIGHT = 20
OBSTACLE_THRESHOLD = .2
OBSTACLE_X_SIZE = 6
OBSTACLE_Y_SIZE = 6
OBSTACLE_X_OFFSET = 0
OBSTACLE_Y_OFFSET = 0

# Real time display of the path planning algorithm
DISPLAY = False
DISPLAY_DELAY = .1
DISPLAY_FREQ = 1
DISPLAY_END = True # Should display for a few seconds at the end of the algorithm (before auto replanning)
WAIT_INPUT = DISPLAY_END and True

# Allow replanning, through user input or programmatically
REPLANNING = False

# For automatic blocked cell generation
BLOCKED_CELLS = 20 # Number of newly blocked cell in one loop 
BLOCKED_CELLS_LOOP = 5 # Number of loops of replanning 

# For the algorithm time duration, average on how many samples
ALGO_DURATION_AVERAGE = 12

# Delay before stopping the algorithm (in seconds)
TIME_OUT = 60