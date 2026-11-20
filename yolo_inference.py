from pathlib import Path
from ultralytics import YOLO

RUNS_DIR = Path(__file__).resolve().parent / "runs"

model = YOLO("models/best.pt")

resutls = model.predict('data/EON.mp4', save=True, project=str(RUNS_DIR))
print(resutls[0])

print('==============================================0')
for box in resutls[0].boxes:
    print(box)
