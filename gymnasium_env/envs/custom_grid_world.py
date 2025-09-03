import logging
import numpy as np
from gymnasium import spaces, Env
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

MAPS = {
    "4x4": ["0000", "0101", "0001", "1000"],
    "8x8": [
        "00000000",
        "00000000",
        "00010000",
        "00000100",
        "00010000",
        "01100010",
        "01001010",
        "00010000",
    ],
}

class CustomGridEnv(Env):

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], 'render_fps': 8}
    FREE: int = 0
    OBSTACLE: int = 1
    MOVES: dict[int,tuple] = {
        0: (-1, 0), #UP
        1: (1, 0),  #DOWN
        2: (0, -1), #LEFT
        3: (0, 1)   #RIGHT
    }

    def __init__(
        self,     
        obstacle_map: str | list[str],
        render_mode: str | None = None,
      ):

      # Env confinguration
      self.obstacles = self.parse_obstacle_map(obstacle_map) #walls
      self.nrow, self.ncol = self.obstacles.shape
      
      self.action_space = spaces.Discrete(len(self.MOVES))
      self.observation_space = spaces.Dict(
          
            {
                "agent": spaces.Box(0, len(obstacle_map) - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, len(obstacle_map) - 1, shape=(2,), dtype=int),
                
            }
            
            )
      self._observation_space = self.observation_space
  
      # Rendering variables
      self.fig = None
      self.render_mode = render_mode
      self.fps = self.metadata['render_fps']
      
    def _get_obs(self): return {"agent": self.agent_xy, "target": self.goal_xy}
    def _get_info(self): return {'position': self.agent_xy}

    
    def parse_obstacle_map(self, obstacle_map):
        map = np.zeros((len(obstacle_map), len(obstacle_map[0])))
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                map[i,j] = int(obstacle_map[i][j])
        return map
    
    
    def reset(
        self, 
        seed: int | None = None, 
        options: dict = dict()
        ) -> tuple:
        super().reset(seed=seed)

        # parse options
        if options == None:
            self.start_xy = (0,0)
            self.goal_xy = (3,3)
        else:
            self.start_xy = options['start']
            self.goal_xy = options['goal']

        # initialise internal vars
        self.agent_xy = self.start_xy
        self.reward = 0
        self.done = False
        self.agent_action = None
        self.n_iter = 0
        
        return self._get_obs(), self._get_info()
    
    def on_goal(self): 
        if self.agent_xy == self.goal_xy: 
            self.reward = 1
            return True 
        else: 
            return False
        
    
    def step(self, action: int):
        self.agent_action = action
        
        x,y = self.agent_xy
        dx,dy = self.MOVES[action]
        
        target_x = x + dx
        target_y = y + dy
        
        if self._observation_space['agent'].contains(np.array([target_x, target_y])) and not self.obstacles[target_x, target_y]:
            self.agent_xy = (target_x,target_y)
            self.done = self.on_goal()
        elif self._observation_space['agent'].contains(np.array([target_x, target_y])) and self.obstacles[target_x, target_y]:
             
             self.agent_xy = (target_x,target_y)
             self.reward = -1
             self.done = True
             self.n_iter += 1
             return self._get_obs(), self.reward, self.done, False, self._get_info()
             
        
        self.n_iter += 1
        
        return self._get_obs(), self.reward, self.done, False, self._get_info()
            
            
        
        
    
                    
                    
            

#gridWorld = CustomGridEnv(obstacle_map=MAPS['4x4'])
#gridWorld.reset(options={'start': (0,0), 'goal': (3,3)})
#gridWorld.step(1)
