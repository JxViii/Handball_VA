import cv2

def read_video(video_path):
  cap = cv2.VideoCapture(video_path)
  frames = []
  while True: 
    found, frame = cap.read() #! return {bool, frame} if bool = false video ended
    if not found:
      break;
    frames.append(frame)
  return frames

def save_video(out_video_frames, out_video_path):
  fourcc = cv2.VideoWriter.fourcc(*'XVID')
  w = out_video_frames[0].shape[1]
  h = out_video_frames[0].shape[0]
  out = cv2.VideoWriter(out_video_path, fourcc, 30, (w,h))
  for frame in out_video_frames:
      out.write(frame)
  out.release()