from math import ceil
import numpy as np
import os
import tempfile
from ...import_utils import *

if is_all_dependency_installed('encoders-video'):
    import librosa
    import soundfile as sf
    from cv2 import cv2
    from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
    from moviepy.video.io.VideoFileClip import VideoFileClip


class FrameSamplingFilter():

    def __init__(self, every=None, hertz=None, top_n=None):
        if every is None and hertz is None and top_n is None:
            raise ValueError("When initializing the FrameSamplingFilter, "
                             "one of the 'every', 'hertz', or 'top_n' must "
                             "be specified.")
        self.every = every
        self.hertz = hertz
        self.top_n = top_n

    def get_audio_sampling_rate(self, filename: str):
        infos = ffmpeg_parse_infos(filename)
        fps = infos.get('audio_fps', 44100)
        if fps == 'unknown':
            fps = 44100
        return fps

    def load_clip(self, filename: str):
        audio_fps = self.get_audio_sampling_rate(filename)
        self.clip = VideoFileClip(filename, audio_fps)

    def initialize_video(self, filename: str):
        self.filename = filename
        self.load_clip(filename)
        self.fps = self.clip.fps
        self.width = self.clip.w
        self.height = self.clip.h
        self.frame_index = range(int(ceil(self.fps * self.clip.duration)))
        self.duration = self.clip.duration
        self.n_frames = len(self.frame_index)

    def get_audio_vector(self, new_sampling_rate: int = 16000):
        fd, fp = tempfile.mkstemp()
        audio = f'{fp}.wav'
        self.clip.audio.to_audiofile(audio)
        data, sampling_rate = sf.read(audio, dtype='float32')
        os.close(fd)
        os.remove(audio)
        return np.array(librosa.resample(data.T, sampling_rate, new_sampling_rate))

    def transform(self, filename: str):
        self.initialize_video(filename)

        if (self.every is not None):
            new_idx = range(self.n_frames)[::self.every]
        elif (self.hertz is not None):
            interval = self.fps / float(self.hertz)
            new_idx = np.arange(0, self.n_frames, interval).astype(int)
            new_idx = list(new_idx)
        elif self.top_n is not None:
            diffs = []
            for i, img in enumerate(range(self.n_frames)):
                if i == 0:
                    last = img
                    continue
                pixel_diffs = cv2.sumElems(cv2.absdiff(
                    self.get_frame(last), self.get_frame(img)))
                diffs.append(sum(pixel_diffs))
                last = img
            new_idx = sorted(range(len(diffs)),
                             key=lambda i: diffs[i],
                             reverse=True)[:self.top_n]

        result = []
        for index in new_idx:
            result.append(self.get_frame(index))
        return result

    def get_frame(self, index: int):
        return self.clip.get_frame(index)

    def iter_frames(self):
        for i, f in enumerate(self.frame_index):
            yield self.get_frame(f)
