import bisect

import muspy
import numpy as np
import torch as th

RESOLUTION = 4  # time steps per quarter note
PITCH_MIN = 21
PITCH_MAX = 108
PITCH_COUNT = PITCH_MAX - PITCH_MIN + 1  # 88 (number of keys on a standard piano)
DURATION_COUNT = 8 * RESOLUTION  # number of note duration bins


class MuspyDataset(muspy.FolderDataset):
    _extension = "mid"

    def read(self, filename: str) -> muspy.Music:
        return muspy.read(filename).adjust_resolution(RESOLUTION)


class TorchDataset(th.utils.data.Dataset):
    def __init__(self, streams: list[np.ndarray], window: int):
        self.window = window
        self.streams = [stream for stream in streams if stream.shape[0] > window]
        self.cumsums = [stream.shape[0] - window for stream in self.streams]

        for i in range(1, len(self.cumsums)):
            self.cumsums[i] += self.cumsums[i - 1]

    def __len__(self):
        return self.cumsums[-1] if len(self.cumsums) > 0 else 0

    def __getitem__(self, index: int):
        sindex = bisect.bisect_right(self.cumsums, index)
        index -= self.cumsums[sindex - 1] if sindex > 0 else 0
        stream = self.streams[sindex]
        return (stream[index:index + self.window], stream[index + self.window])


class DataFormat:
    def encode(self, music: muspy.Music) -> np.ndarray:
        raise NotImplementedError()

    def decode(self, data: np.ndarray) -> muspy.Music:
        raise NotImplementedError()


class PianorollFormat(DataFormat):
    def encode(self, music: muspy.Music) -> np.ndarray:
        data = muspy.to_pianoroll_representation(music, encode_velocity=False, dtype=np.float32)

        # Make sure repeated notes aren't "fused" together by inserting a gap before each one
        for track in music.tracks:
            for note in track.notes:
                if note.time > 0:
                    data[note.time - 1, note.pitch] = 0.0

        return data[:, PITCH_MIN:PITCH_MAX + 1]

    def decode(self, data: np.ndarray) -> muspy.Music:
        return muspy.from_pianoroll_representation(
            np.pad(data > 0.0, ((0, 0), (PITCH_MIN, 127 - PITCH_MAX))), RESOLUTION, encode_velocity=False
        )


class DurationPianorollFormat(DataFormat):
    def __init__(self, fade: bool = False, normalize: bool = True):
        self.fade = fade  # fade note durations over multiple timesteps
        self.normalize = normalize  # normalize durations to [0.0, 1.0]

    def encode(self, music: muspy.Music) -> np.ndarray:
        notes: list[tuple[int, int, int]] = []  # [(time, pitch, duration), ...]
        length = 0

        for track in music.tracks:
            for note in track.notes:
                length = max(length, note.end)
                time, duration = note.time, note.duration

                # Split notes into pieces no longer than DURATION_COUNT
                while duration > 0:
                    duration_split = min(duration, DURATION_COUNT)
                    duration -= duration_split
                    notes.append((time, note.pitch, duration_split))
                    time += duration_split

        notes.sort()
        data = np.zeros((length, 128), dtype=np.float32)

        for time, pitch, duration in notes:
            if self.fade:
                data[time:time + duration, pitch] = np.arange(duration, 0, -1)
            else:
                data[time, pitch] = duration

        # Normalize to [0.0, 1.0] with data points in the centers of DURATION_COUNT + 1 bins
        if self.normalize:
            data += 0.5
            data /= DURATION_COUNT + 1

        return data[:, PITCH_MIN:PITCH_MAX + 1]

    def decode(self, data: np.ndarray) -> muspy.Music:
        if self.normalize:
            data = np.floor(data * (self.limit + 1))

        notes: list[tuple[int, int, int]] = []  # [(time, pitch, duration), ...]
        for time, pitch in np.argwhere(np.diff(data, axis=0, prepend=0.0) > 0.0):
            notes.append((time, pitch + PITCH_MIN, int(data[time, pitch])))

        return muspy.Music(
            resolution=RESOLUTION, tracks=[muspy.Track(
                notes=[muspy.Note(time, pitch, duration) for time, pitch, duration in sorted(notes)]
            )]
        )


def load_folder(path: str, window: int, fmt: DataFormat) -> TorchDataset:
    return TorchDataset(map(fmt.encode, MuspyDataset(path)), window)

