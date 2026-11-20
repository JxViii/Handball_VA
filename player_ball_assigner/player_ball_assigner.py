import sys
sys.path.append('../')
from utils import get_center_bbox, measure_distance

# ! Player-Ball -> pb

class PlayerBallAssigner:
  def __init__(self):
    self.max_pb_distance = 70
    
  def assign_b_to_p(self, players, ball_bbox):
    ball_pos = get_center_bbox(ball_bbox)
    
    min_distance = 9999999
    pv = -1 # ! Assigned player
    
    for player_id, player in players.items():
      player_bbox = player['bbox']
      
      distance_l = measure_distance((player_bbox[0], player_bbox[1]), ball_pos)
      distance_r = measure_distance((player_bbox[2], player_bbox[1]), ball_pos)
      distance = min(distance_l, distance_r)
      
      if distance < self.max_pb_distance:
          if distance < min_distance:
            min_distance = distance
            pv = player_id
            
    return pv
    
      
      
    
  
  
    