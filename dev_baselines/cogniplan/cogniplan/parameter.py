# saving path
FOLDER_NAME = 'cogniplan_exp_pred7_test'
model_path = f'checkpoints/{FOLDER_NAME}'
log_path = f'{model_path}/log'
gif_path = f'{model_path}/gif'

# predictor settings
generator_path = f'checkpoints/wgan_inpainting'
N_GEN_SAMPLE = 4  # how many samples do you want to generate

# save training data
SUMMARY_WINDOW = 16  # how many training steps before writing data to tensorboard
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 2000  # how many episodes before saving a gif

# map and planning resolution
CELL_SIZE = 0.4  # meter, your map resolution
NODE_RESOLUTION = 2.0  # meter, your node resolution
FRONTIER_CELL_SIZE = 1 * CELL_SIZE  # do you want to downsample the frontiers

# map representation (matches ROS OccupancyGrid convention)
FREE = 0      # value of free cells in the map
OCCUPIED = 100  # value of obstacle cells in the map
UNKNOWN = -1  # value of unknown cells in the map

# sensor and utility range
SENSOR_RANGE = 20  # meter
UTILITY_RANGE = 0.5 * SENSOR_RANGE  # consider frontiers within this range as observable
MIN_UTILITY = 3  # ignore the utility if observable frontiers are less than this value

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # nodes outside this range will not be affected by current measurements

# graph update thresholds
THR_GRAPH_HARD_UPDATE = 8  # meter, node and edges within this range will be fully re-evaluated
CLUSTER_RANGE = 10  # meter, frontiers will be clustered based on this range
THR_NEXT_WAYPOINT = 5  # meter, rarefied graph waypoint distance threshold

# training parameters
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 64
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 16  # how many threads does your CPU have

# network parameters
NODE_INPUT_DIM = 6
EMBEDDING_DIM = 128

# Graph parameters
K_SIZE = 25  # the number of neighboring nodes, fixed
NODE_PADDING_SIZE = 650  # the number of nodes will be padded to this value, need it for batch training

# GPU usage
USE_GPU = False  # do you want to collect training data using GPUs (better not)
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 0  # 0 unless you want to collect data using GPUs

USE_WANDB = False
