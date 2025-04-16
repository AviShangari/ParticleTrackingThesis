from moviepy import VideoFileClip, clips_array

# Load videos (force no audio to avoid codec errors)
clip1 = VideoFileClip("./Tracking/vid10.mp4")
clip2 = VideoFileClip("./Tracking/Output/Skeletal_Tracking/skeleton_tracked_video10-2.mp4")

# Resize to same height
min_height = min(clip1.h, clip2.h)
clip1_resized = clip1.resized(height=min_height)
clip2_resized = clip2.resized(height=min_height)

# Combine side-by-side
final_clip = clips_array([[clip1_resized, clip2_resized]])

# Write output video using compatible codec
final_clip.write_videofile("./Tracking/side_by_side10.mp4", codec="libx264", audio=False, fps=30)
