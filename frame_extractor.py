from moviepy.editor import VideoFileClip
import numpy as np
import os
import shutil

def check_folder(path) -> None:
    '''
        checks for the folder given as the path. if not exists creates one. 
        if it already exists deletes all the content in the folder.
        path = path of the folder
    '''
    if os.path.isdir(path):
        shutil.rmtree(path)
    print("PATH ", path)
    os.mkdir(path) #clearing the elements in the folder

def save_frames(video_file, output_folder="video_images", frames_per_second = 10, max_frames = 1000) -> None:
    """
        segregates the frames from the given video file and saves in the given output folder.

        video_file = video file path
        output_folder = folder to where to save the frames
        frames_per_second = number of frames to be used as fps of the video
        max_frames = total number of frames that can be extracted from the entire video
    """
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