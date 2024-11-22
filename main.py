import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
from datetime import datetime
from utils import read_video, save_video
from trackers import Tracker
from view_transformer import ViewTransformer
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import matplotlib.pyplot as plt
from fpdf import FPDF

class VideoProcessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Football Analysis System")
        self.geometry("900x600")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Frame for controls
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        self.upload_button = ctk.CTkButton(self.controls_frame, text="Upload Video", command=self.upload_video)
        self.upload_button.pack(side=tk.LEFT, padx=10)

        self.process_button = ctk.CTkButton(self.controls_frame, text="Process Video", command=self.process_video)
        self.process_button.pack(side=tk.LEFT, padx=10)

        self.play_button = ctk.CTkButton(self.controls_frame, text="Play Video", command=self.play_video)
        self.play_button.pack(side=tk.LEFT, padx=10)

        self.stats_button = ctk.CTkButton(self.controls_frame, text="Generate Statistics", command=self.generate_statistics)
        self.stats_button.pack(side=tk.LEFT, padx=10)

        self.exit_button = ctk.CTkButton(self.controls_frame, text="Exit", command=self.on_closing)
        self.exit_button.pack(side=tk.RIGHT, padx=10)

        # Label for displaying status and video
        self.status_video_label = ctk.CTkLabel(self, text="AI FOOTBALL ANALYSIS SYSTEM", 
                                              font=ctk.CTkFont(size=36, weight="bold"))
        self.status_video_label.pack(expand=True, fill=tk.BOTH)

        # Progress bar
        self.progress = ttk.Progressbar(self, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)

        self.video_path = None
        self.tracks = None
        self.output_video_path = None
        self.total_distance_covered = {}
        self.cap = None
        self.stop_playback = False

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file")
        else:
            self.status_video_label.configure(text="Video Uploaded Successfully")
            self.processing_label_visible = False
            self.status_video_label.pack(expand=True, fill=tk.BOTH)

    def process_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please upload a video first")
            return

        self.progress.start()
        threading.Thread(target=self.run_analysis).start()

    def run_analysis(self):
        self.display_message("Processing video...")
        
        # Read Video
        video_frames = read_video(self.video_path)

        # Initialize Tracker
        tracker = Tracker('models/best.pt')
        self.tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)
        tracker.add_position_to_tracks(self.tracks)

        # Camera movement estimator
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=False)
        camera_movement_estimator.add_adjust_positions_to_tracks(self.tracks, camera_movement_per_frame)

        # View Transformer
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(self.tracks)

        # Interpolate Ball Positions
        self.tracks["ball"] = tracker.interpolate_ball_positions(self.tracks["ball"])

        # Speed and distance estimator
        speed_and_distance_estimator = SpeedAndDistance_Estimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(self.tracks)
        self.total_distance_covered = speed_and_distance_estimator.total_distance

        # Assign Player Teams
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], self.tracks['players'][0])

        for frame_num, player_track in enumerate(self.tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                self.tracks['players'][frame_num][player_id]['team'] = team
                self.tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

        # Assign Ball Acquisition
        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        for frame_num, player_track in enumerate(self.tracks['players']):
            ball_bbox = self.tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                self.tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(self.tracks['players'][frame_num][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1])
        team_ball_control = np.array(team_ball_control)

        # Draw output
        output_video_frames = tracker.draw_annotations(video_frames, self.tracks, team_ball_control)
        output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, self.tracks)

        # Stop the progress bar
        self.progress.stop()
        self.display_message("Video processing complete!")

        # Save the processed video
        unique_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_video_path = os.path.join("output_videos", f"processed_video_{unique_name}.avi")
        save_video(output_video_frames, self.output_video_path)

        self.display_message("Processed video saved!")

    def play_video(self):
        if not self.output_video_path:
            messagebox.showerror("Error", "Please process a video first")
            return

        self.cap = cv2.VideoCapture(self.output_video_path)
        self.stop_playback = False
        self.play_video_frames()

    def play_video_frames(self):
        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror("Error", "Error in opening the video file.")
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)
            self.status_video_label.configure(image=frame)
            self.status_video_label.image = frame
            if not self.stop_playback:
                self.status_video_label.after(10, self.play_video_frames)
        else:
            self.cap.release()
            self.status_video_label.configure(image=None, text="Video playback finished")

    def generate_statistics(self):
        if not self.tracks:
            messagebox.showerror("Error", "Please process a video first")
            return

        # Example: Plot total distance covered by each player
        for object, object_distances in self.total_distance_covered.items():
            plt.figure(figsize=(10, 6))
            plt.title(f"Total Distance Covered by {object.capitalize()}s")
            plt.bar(object_distances.keys(), object_distances.values())
            plt.xlabel("Player ID")
            plt.ylabel("Distance Covered (m)")
            plt.savefig(f"stats_{object}_distance.png")
            plt.close()

        # Generate a PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Football Analysis Report", ln=True, align="C")

        for object, object_distances in self.total_distance_covered.items():
            pdf.cell(200, 10, txt=f"Total Distance Covered by {object.capitalize()}s:", ln=True)
            for player_id, distance in object_distances.items():
                pdf.cell(200, 10, txt=f"Player {player_id}: {distance:.2f} meters", ln=True)

            # Add the plot to the PDF
            pdf.image(f"stats_{object}_distance.png", x=10, y=pdf.get_y(), w=180)
            pdf.ln(85)  # move to the next line after the image

        pdf.output(f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

        messagebox.showinfo("Success", "Statistics and PDF report generated!")

    def display_message(self, message):
        self.status_video_label.configure(text=message)

    def on_closing(self):
        self.stop_playback = True
        self.quit()
        self.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VideoProcessorApp()
    app.mainloop()