from moviepy.editor import VideoFileClip
import numpy as np
import os
import shutil

def check_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path) #clearing the elements in the folder

def save_frames(video_file, output_folder="video_images", frames_per_second = 10, max_frames = 1000):
    video = VideoFileClip(video_file)
    check_folder(output_folder)

    saving_frames_per_second = min(video.fps, frames_per_second)
    # if SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
    step = 1 / video.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second

    filename = 1
    for current_duration in np.arange(0, video.duration, step):
        if filename > max_frames:
            break
        video.save_frame(os.path.join(output_folder, str(filename)+".jpg"), current_duration)
        filename += 1
    print("successfully extracted all the frames")


if __name__ == '__main__':
    save_frames("..\\demo.mkv","..\\video_images",3,100)