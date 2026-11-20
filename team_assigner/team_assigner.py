import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
  def __init__(self):
    self.team_colors = {}
    self.kmeans = None
    self.player_team_dict = {}
    

  def get_player_color(self, frame, bbox):
    # ! bbox = [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, bbox)
    h = y2 - y1
    w = x2 - x1

    # ! Torso box: 15-50 % of the height (no head, no shorts),
    # ! middle half of the width (very little court)
    torso = frame[
      y1 + int(0.15 * h) : y1 + int(0.5 * h),  # ! y
      x1 + int(0.25 * w) : x2 - int(0.25 * w)  # ! x
    ]

    # if torso.size < 30: 
    #   return None

    # ! Median color in Lab: ignores the number, trim and the bit of court left
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab, axis=0)

  def assign_team_color(self, frame, player_detections):
    players_colors = []
    for _, player_detection in player_detections.items():
      bbox = player_detection["bbox"]
      player_color = self.get_player_color(frame, bbox)
      if player_color is not None:
        players_colors.append(player_color)

    if len(players_colors) < 2:
      raise ValueError(f"Need at least 2 player colors to make 2 teams, got {len(players_colors)}")

    kmeans = KMeans(n_clusters = 2, init="k-means++", n_init=1)
    kmeans.fit(players_colors)
    
    self.kmeans = kmeans
    
    # ! Centers are Lab (what k-means works in), but OpenCV draws in BGR -> convert for drawing
    self.team_colors[1] = self.lab_to_bgr(kmeans.cluster_centers_[0])
    self.team_colors[2] = self.lab_to_bgr(kmeans.cluster_centers_[1])

  def lab_to_bgr(self, lab_color):
    lab_pixel = np.uint8([[lab_color]])  # ! 1x1 image so cvtColor accepts it
    return tuple(int(c) for c in cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)[0, 0])
    
    
  def get_player_team(self, frame, player_bbox, player_id):
    if player_id in self.player_team_dict:
      return self.player_team_dict[player_id]
    
    player_color = self.get_player_color(frame, player_bbox)
    team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
    team_id+=1 # ! team_id -> [0,1]
    
    self.player_team_dict[player_id] = team_id
    
    return team_id
    
      
  