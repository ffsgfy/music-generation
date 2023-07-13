import bisect

import muspy
import numpy as np
import torch as th

RESOLUTION = 8  # time steps per quarter note


class MuspyDataset(muspy.FolderDataset):
    _extension = "mid"

    def read(self, filename: str) -> muspy.Music:
        return muspy.read(filename).adjust_resolution(RESOLUTION)


class TorchDataset(th.utils.data.Dataset):
    def __init__(self, rolls: list[np.ndarray], window: int):
        self.window = window
        self.rolls = [roll for roll in rolls if roll.shape[0] > window]
        self.cumsums = [roll.shape[0] - window for roll in self.rolls]

        for i in range(1, len(self.cumsums)):
            self.cumsums[i] += self.cumsums[i - 1]

    def __len__(self):
        return self.cumsums[-1] if len(self.cumsums) > 0 else 0

    def __getitem__(self, index: int):
        rindex = bisect.bisect_right(self.cumsums, index)
        index -= self.cumsums[rindex - 1] if rindex > 0 else 0
        roll = self.rolls[rindex]
        return (roll[index:index + self.window], roll[index + self.window])


def pianoroll_encode(music: muspy.Music) -> np.ndarray:
    roll = muspy.to_pianoroll_representation(music, encode_velocity=False)

    # Make sure repeated notes aren't "fused" together by inserting a gap before each one
    for track in music.tracks:
        for note in track.notes:
            if note.time > 0:
                roll[note.time - 1, note.pitch] = False

    return roll.T[21:109].T.astype(np.float32)


def pianoroll_decode(roll: np.ndarray) -> muspy.Music:
    return muspy.from_pianoroll_representation(
        np.pad(roll > 0.0, ((0, 0), (21, 19))), RESOLUTION, encode_velocity=False
    )


def load_folder(root: str, window: int) -> TorchDataset:
    return TorchDataset(map(pianoroll_encode, MuspyDataset(root, convert=True)), window)

