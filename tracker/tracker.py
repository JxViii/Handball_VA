from ultralytics import YOLO
import supervision as sv
from trackers import ByteTrackTracker
import cv2
import pickle
import os
import sys
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import COLORS, get_center_bbox, get_width_bbox


class Tracker:

  def __init__(self, model_path, ball_model_path=None):
    self.model = YOLO(model_path)
    # ! Optional dedicated ball model; if None, the ball comes from the main model
    self.ball_model = YOLO(ball_model_path) if ball_model_path else None
    self.tracker = ByteTrackTracker()
    
  # ! Split the detections into segments of physically plausible motion
  # ! (jump <= max_speed px per frame of gap) and drop very short segments:
  # ! false positives (tape, shoes, lights...) appear for 1-2 frames, the real ball persists.
  # ! Unlike comparing against a single "last valid" detection, this can't get stuck.
  def remove_ball_noise(self, ball_positions, max_speed=80, min_segment=3):
    frames, centers = [], []
    for i, x in enumerate(ball_positions):
      bbox = x.get(1, {}).get("bbox", [])
      if len(bbox):
        frames.append(i)
        centers.append(np.array(get_center_bbox(bbox)))

    segments, cur = [], [0] if frames else []
    for k in range(1, len(frames)):
      gap = frames[k] - frames[k - 1]
      if np.linalg.norm(centers[k] - centers[k - 1]) > max_speed * gap:
        segments.append(cur)
        cur = []
      cur.append(k)
    if cur:
      segments.append(cur)

    for seg in segments:
      if len(seg) < min_segment:
        for k in seg:
          ball_positions[frames[k]] = {}

    return ball_positions

  # ! Fill only short gaps (<= max_gap frames) between two detections.
  # ! Long gaps (ball held/hidden/out of frame) and the start/end of the video stay empty
  # ! instead of drawing a ball sliding in a straight line or frozen in place.
  def interpolate_ball_positions(self, ball_positions, max_gap=15):
    rows = [x.get(1, {}).get("bbox", [np.nan] * 4) for x in ball_positions]
    df = pd.DataFrame(rows, columns=['x1', 'y1', 'x2', 'y2'])

    missing = df['x1'].isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform('size')

    filled = df.interpolate(limit_area='inside')
    filled[missing & (run_len > max_gap)] = np.nan

    return [{1: {"bbox": r}} if not np.isnan(r[0]) else {} for r in filled.to_numpy().tolist()]

  def detect_frames(self, frames, model):
    batch_size = 20
    detections = []
    for i in range(0, len(frames), batch_size):
      detections_batch = model.predict(frames[i:i + batch_size], conf=0.1)
      detections += detections_batch
    return detections

  def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

    # ! Store results so no additional work
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
      with open(stub_path, 'rb') as f:
        tracks = pickle.load(f)
      return tracks

    detections = self.detect_frames(frames, self.model)
    ball_detections = self.detect_frames(frames, self.ball_model) if self.ball_model else None

    tracks = {
      "players": [],
      "referees": [],
      "gks": [],
      "ball": []
    }

    for frame_num, detection in enumerate(detections):
      cls_names = detection.names
      cls_names_inverse = {v: k for k, v in cls_names.items()}

      # * Convert to sv Detection format
      detection_sv = sv.Detections.from_ultralytics(detection)

      # * Ball source: dedicated ball model if given, otherwise the main model
      if ball_detections is not None:
        ball_sv = sv.Detections.from_ultralytics(ball_detections[frame_num])
        ball_cls_id = {v.lower(): k for k, v in ball_detections[frame_num].names.items()}['ball']
        # ! Drop the main model's balls so they don't reach the tracker
        detection_sv = detection_sv[detection_sv.class_id != cls_names_inverse['ball']]
      else:
        ball_sv = detection_sv
        ball_cls_id = cls_names_inverse['ball']

      # * Track Object
      detection_w_tracks = self.tracker.update(detection_sv)

      tracks["players"].append({})
      tracks["gks"].append({})
      tracks["referees"].append({})
      tracks["ball"].append({})

      for frame_detection in detection_w_tracks:
        bbox = frame_detection[0].tolist()
        cls_id = frame_detection[3]
        track_id = int(frame_detection[4])

        # ! -1 = detection the tracker hasn't confirmed yet (new or low confidence), skip it
        if track_id == -1:
          continue

        if cls_id == cls_names_inverse['players']:
          tracks["players"][frame_num][track_id] = {"bbox": bbox}

        if cls_id == cls_names_inverse['referees']:
          tracks["referees"][frame_num][track_id] = {"bbox": bbox}

        if cls_id == cls_names_inverse['goalkeeper']:
          tracks["gks"][frame_num][track_id] = {"bbox": bbox}

      # ! Only the ball (not tracked, so it always uses key 1)
      # ! Several "balls" can appear (white tape/bandages), keep only the most confident one
      ball_mask = ball_sv.class_id == ball_cls_id
      if ball_mask.any():
        best = np.argmax(np.where(ball_mask, ball_sv.confidence, -1))
        tracks["ball"][frame_num][1] = {"bbox": ball_sv.xyxy[best].tolist()}

    # ! Stub = Optional (saved once, after every frame is processed)
    if stub_path is not None:
      with open(stub_path, 'wb') as f:
        pickle.dump(tracks, f)

    return tracks

  def draw_ellipse(self, frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center, _ = get_center_bbox(bbox)
    w = get_width_bbox(bbox)

    cv2.ellipse(
      frame,
      center=(int(x_center), y2),
      axes=(int(w), int(0.5 * w)),
      angle=0,
      startAngle=-45,
      endAngle=235,
      color=color,
      thickness=2,
      lineType=cv2.LINE_4
    )
    
    rect_w = 40
    rect_h = 20
    x1_rect = x_center - rect_w // 2
    x2_rect = x_center + rect_w // 2
    y1_rect = (y2 - rect_h // 2) + 15
    y2_rect = (y2 + rect_h // 2) + 15
    
    if track_id is not None:
      cv2.rectangle(frame,
                    (x1_rect, y1_rect),
                    (x2_rect, y2_rect),
                    color,
                    cv2.FILLED)
      
      x1_text = x1_rect + 12
      if track_id > 99:
        x1_text -= 10

      # ! Perceived brightness of the box color (BGR): dark box -> white text, light box -> black text
      b, g, r = color
      brightness = 0.114 * b + 0.587 * g + 0.299 * r
      text_color = COLORS["white"] if brightness < 128 else COLORS["black"]

      cv2.putText(
        frame,
        f"{track_id}",
        (int(x1_text), int(y1_rect + 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        2
      )

    return frame
  
  def draw_triangle(self, frame, bbox, color):
    y = int(bbox[1])
    x, _ = get_center_bbox(bbox)
    h = 20
    d = 10
    
    tr_pts = np.array([
      [x,y],
      [x - d, y - h],
      [x + d, y - h]
    ])
    
    cv2.drawContours(frame,
                     [tr_pts],
                     0,
                     color,
                     cv2.FILLED)
    
    cv2.drawContours(frame,
                     [tr_pts],
                     0,
                     COLORS["white"],
                     2)    
    
    return frame

  def draw_annotations(self, video_frames, tracks):
    out_video_frames = []
    for frame_num, frame in enumerate(video_frames):
      frame = frame.copy()

      player_dict = tracks["players"][frame_num]
      gk_dict = tracks["gks"][frame_num]
      referee_dict = tracks["referees"][frame_num]
      ball_dict = tracks["ball"][frame_num]

      # * Draw Players
      for track_id, player in player_dict.items():
        color = player.get("team-color", COLORS["black"])
        frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
        
        if player.get("has_ball", False):
          frame = self.draw_triangle(frame, player["bbox"], COLORS["red"])
        
      # * Draw GKS
      for track_id, gk in gk_dict.items():
        frame = self.draw_ellipse(frame, gk["bbox"], COLORS["lime"], track_id)
        
      # * Draw Players
      for _, ref in referee_dict.items():
        frame = self.draw_ellipse(frame, ref["bbox"], COLORS["orange"])
        
      # * Draw Ball
      for track_id, ball in ball_dict.items():
        frame = self.draw_triangle(frame, ball["bbox"], COLORS["magenta"])

      out_video_frames.append(frame)

    return out_video_frames

