import enum
import bisect

import muspy
import numpy as np
import torch as th

RESOLUTION = 4  # time steps per quarter note
PITCH_MIN = 21
PITCH_MAX = 108
PITCH_COUNT = PITCH_MAX - PITCH_MIN + 1  # 88 (number of keys on a standard piano)
DURATION_COUNT = 8 * RESOLUTION  # number of note duration bins
OFFSET_COUNT = 8 * RESOLUTION  # number of bar/note offset bins


class MuspyDataset(muspy.FolderDataset):
    _extension = "mid"

    def read(self, filename: str) -> muspy.Music:
        return muspy.read(filename).adjust_resolution(RESOLUTION)


class TorchDataset(th.utils.data.Dataset):
    def __init__(self, streams: list[np.ndarray], window: int):
        self.window = window
        self.streams = [stream for stream in streams if stream.shape[0] >= window]
        self.cumsums = [stream.shape[0] - window + 1 for stream in self.streams]

        for i in range(1, len(self.cumsums)):
            self.cumsums[i] += self.cumsums[i - 1]

    def __len__(self) -> int:
        return self.cumsums[-1] if len(self.cumsums) > 0 else 0

    def __getitem__(self, index: int) -> np.ndarray:
        sindex = bisect.bisect_right(self.cumsums, index)
        index -= self.cumsums[sindex - 1] if sindex > 0 else 0
        stream = self.streams[sindex]
        return stream[index:index + self.window]


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


class EventFormat(DataFormat):
    class EventKind(enum.IntEnum):
        BAR = 0
        OFFSET = 1
        PITCH = 2
        DURATION = 3

    EVENT_COUNT = 1 + OFFSET_COUNT + PITCH_COUNT + DURATION_COUNT
    EVENT_OFFSETS = (0, 1, 1 + OFFSET_COUNT, 1 + OFFSET_COUNT + PITCH_COUNT, EVENT_COUNT)

    def encode(self, music: muspy.Music) -> np.ndarray:
        music = music.infer_barlines()
        notes: list[tuple[int, int, int]] = []  # [(time, pitch, duration), ...]

        for track in music.tracks:
            for note in track.notes:
                time, duration = note.time, note.duration
                pitch = note.pitch - PITCH_MIN
                assert (pitch >= 0) and (pitch < PITCH_COUNT)

                # Split notes into pieces no longer than DURATION_COUNT
                while duration > 0:
                    duration_split = min(duration, DURATION_COUNT)
                    duration -= duration_split
                    notes.append((time, pitch, duration_split))
                    time += duration_split

        # Split bars into pieces no longer than OFFSET_COUNT - 1
        time_base = 0
        for time in sorted(barline.time for barline in music.barlines):
            while time > time_base:
                time_base += min(time - time_base, OFFSET_COUNT - 1)
                notes.append((time_base, -1, 0))  # add fake notes with pitch = -1 to represent barlines

        notes.sort()
        events: list[int] = []
        time_base = 0

        for time, pitch, duration in notes:
            offset = time - time_base
            assert offset < OFFSET_COUNT

            events.append(self.EVENT_OFFSETS[self.EventKind.OFFSET] + offset)
            if pitch >= 0:
                events.append(self.EVENT_OFFSETS[self.EventKind.PITCH] + pitch)
                events.append(self.EVENT_OFFSETS[self.EventKind.DURATION] + duration - 1)
            else:
                events.append(self.EVENT_OFFSETS[self.EventKind.BAR])
                time_base = time

        return np.array(events, dtype=np.int64)

    def decode(self, data: np.ndarray) -> muspy.Music:
        notes: list[tuple[int, int, int]] = []  # [(time, pitch, duration), ...]
        time_base = 0
        pitch = -1
        offset = -1

        for event in data:
            if event >= self.EVENT_OFFSETS[self.EventKind.DURATION]:
                if (pitch >= 0) and (offset >= 0):
                    duration = event - self.EVENT_OFFSETS[self.EventKind.DURATION]
                    notes.append((time_base + offset, pitch + PITCH_MIN, duration + 1))
                pitch = -1
                offset = -1
            elif event >= self.EVENT_OFFSETS[self.EventKind.PITCH]:
                if offset >= 0:
                    pitch = event - self.EVENT_OFFSETS[self.EventKind.PITCH]
            elif event >= self.EVENT_OFFSETS[self.EventKind.OFFSET]:
                pitch = -1
                offset = event - self.EVENT_OFFSETS[self.EventKind.OFFSET]
            else:
                if offset >= 0:
                    time_base += offset
                pitch = -1
                offset = -1

        return muspy.Music(
            resolution=RESOLUTION, tracks=[muspy.Track(
                notes=[muspy.Note(time, pitch, duration) for time, pitch, duration in sorted(notes)]
            )]
        )

    @classmethod
    def next_mask(cls, event: int) -> np.ndarray:
        result = np.zeros(cls.EVENT_COUNT, dtype=bool)

        def mask_kind(kind: cls.EventKind):
            nonlocal cls, result
            result[cls.EVENT_OFFSETS[kind]:cls.EVENT_OFFSETS[kind + 1]] = True

        if event >= cls.EVENT_OFFSETS[cls.EventKind.DURATION]:
            # DURATION is followed by OFFSET
            mask_kind(cls.EventKind.OFFSET)
        elif event >= cls.EVENT_OFFSETS[cls.EventKind.PITCH]:
            # PITCH is followed by DURATION
            mask_kind(cls.EventKind.DURATION)
        elif event >= cls.EVENT_OFFSETS[cls.EventKind.OFFSET]:
            # OFFSET is followed by BAR or PITCH
            mask_kind(cls.EventKind.BAR)
            mask_kind(cls.EventKind.PITCH)
        else:
            # BAR is followed by OFFSET
            mask_kind(cls.EventKind.OFFSET)

        return result


def load_folder(path: str, window: int, fmt: DataFormat) -> TorchDataset:
    return TorchDataset(map(fmt.encode, MuspyDataset(path)), window)

