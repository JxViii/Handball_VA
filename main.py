from utils import read_video, save_video
from tracker import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

# team_name = "CAS-LOG"
video_path = "data/TLV-ADE.mp4"
stub_path = "stubs/TLV-ADE.pkl"
out_path = "out/TLV-ADE.avi"

def main():
    # * Read Video
    video_frames = read_video(video_path)    

    tracker = Tracker('models/1280_yolo26m/best.pt',
                      ball_model_path='models/1280_yolo26m_ball/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=stub_path)
    
    # * Interpolate Ball Positions
    
    tracks["ball"] = tracker.remove_ball_noise(tracks["ball"])
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # * Assign Player Teams
    
    TA = TeamAssigner()
    
    # ! Frame 0 is empty (tracks need 2 frames to be confirmed),
    # ! so use the first frame with at least 2 players
    first_frame = next(i for i, players in enumerate(tracks["players"]) if len(players) >= 2)
    TA.assign_team_color(video_frames[first_frame], tracks["players"][first_frame])
    
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = TA.get_player_team(video_frames[frame_num],
                                      track["bbox"],
                                      player_id)
            tracks["players"][frame_num][player_id]['team'] = team
            tracks["players"][frame_num][player_id]['team-color'] = TA.team_colors[team]
            
    # * Assign ball aquisition
    
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks["players"]):
        ball = tracks["ball"][frame_num].get(1)
        if ball is None:  # ! No ball in this frame
            continue
        pv = player_assigner.assign_b_to_p(player_track, ball["bbox"]) # ! Assigned player
        
        if pv != -1:
            tracks["players"][frame_num][pv]["has_ball"] = True
    
    
    # * Draw output
    
    out_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    # * Save Video
    save_video(out_video_frames, out_path)

if __name__ == '__main__':
    main()
    
    
#     ! Save Cropped Image of a player (frame 0)

#     for track_id, player in tracks["players"][0].items():
#     bbox = player["bbox"]
#     frame = video_frames[0]

#     # ! crop bbox from frame
#     # ! bbox = {x1,y1,x2,y2}
#     crop_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

#     # ! save cropped
#     cv2.imwrite(f'out/{team_name}/crop_img.jpg', crop_img)
#     break