import math
from typing import Callable, TypeVar

import torch as th
import numpy as np
from tqdm import tqdm

import dataset


class ModelBase(th.nn.Module):
    def __init__(self):
        super().__init__()

        self.data_fmt = dataset.EventFormat()


ModelT = TypeVar("ModelT", bound=ModelBase)
BatchT = TypeVar("BatchT")


def train(
    model: ModelT,
    # Model identifier used in state filenames
    model_id: str,
    # Processes a batch of inputs
    batch_fn: Callable[[th.Tensor], BatchT],
    # Calculates and returns the loss
    loss_fn: Callable[[ModelT, BatchT], th.Tensor],
    # Optional path to state from which to continue training
    state_path: str | None,
    # Tuple = separate train/test epoch sizes; float = fraction of dataset
    epoch_size: int | float | tuple[int | float, int | float],
    # Optional first epoch number override
    epoch_start: int | None,
    # Maximum number of epochs
    epoch_limit: int,
    # Dataset window size
    window: int,
    # Tuple = separate train/test batch sizes
    batch_size: int | tuple[int, int],
    # Initial learning rate (overrides state)
    lr: float,
    # Learning rate exponential decay factor
    lr_decay: float,
    # Minimum learning rate
    lr_limit: float
):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = th.optim.Adamax(model.parameters(), lr=lr)

    if state_path is not None:
        state = th.load(state_path)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if epoch_start is None:
            epoch_start = state["last_epoch"] + 1

    # NOTE: initial_lr saved in state is overridden
    for group in optimizer.param_groups:
        group["initial_lr"] = lr

    if epoch_start is None:
        epoch_start = 0

    if isinstance(batch_size, (tuple, list)):
        batch_size_train, batch_size_test = batch_size
    else:
        batch_size_train, batch_size_test = batch_size, batch_size

    scheduler = th.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: max(lr_decay ** epoch, lr_limit), epoch_start - 1
    )

    train_ds = dataset.load_folder("input/ff/train/", window=window, fmt=model.data_fmt)
    train_dl = th.utils.data.DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
    test_ds = dataset.load_folder("input/ff/test/", window=window, fmt=model.data_fmt)
    test_dl = th.utils.data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=True)

    train_total = (len(train_ds) + batch_size_train - 1) // batch_size_train
    test_total = (len(test_ds) + batch_size_test - 1) // batch_size_test

    if isinstance(epoch_size, (tuple, list)):
        epoch_size_train, epoch_size_test = epoch_size
    else:
        epoch_size_train, epoch_size_test = epoch_size, epoch_size

    if isinstance(epoch_size_train, float):
        epoch_size_train = int(train_total * epoch_size_train)

    if isinstance(epoch_size_test, float):
        epoch_size_test = int(test_total * epoch_size_test)

    epoch_size_train = min(epoch_size_train, train_total)
    epoch_size_test = min(epoch_size_test, test_total)

    for epoch in range(epoch_start, epoch_limit):
        print(f"\nModel {model_id}, epoch {epoch}")

        model.train()
        loss_sum = 0.0
        for i, batch in enumerate(tqdm_it := tqdm(train_dl, "Training", epoch_size_train, dynamic_ncols=True)):
            if i >= epoch_size_train:
                break

            batch = batch_fn(batch.to(device))
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss)
            tqdm_it.set_description(f"Training ({loss_sum / (i + 1):.05f})")

        # This is to allow testing of models without training
        if epoch_size_train > 0:
            scheduler.step()
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "last_epoch": epoch
            }
            state_path = f"models/{model_id}.{epoch}.pth"
            th.save(state, state_path)
            print(f"State saved to {state_path}")

        model.eval()
        loss_sum = 0.0
        with th.no_grad():
            for i, batch in enumerate(tqdm_it := tqdm(test_dl, "Testing", epoch_size_test, dynamic_ncols=True)):
                if i >= epoch_size_test:
                    break

                batch = batch_fn(batch.to(device))
                loss = loss_fn(model, batch)
                loss_sum += float(loss)
                tqdm_it.set_description(f"Testing ({loss_sum / (i + 1):.05f})")


def cleanup(model: ModelBase, data: th.Tensor, threshold: float) -> th.Tensor:
    probs = th.nn.functional.softmax(model(data), dim=-1)
    maxima = probs.max(dim=-1)
    diffs = maxima.values - probs[th.arange(probs.size(0)), data]
    nmask = diffs < threshold
    result = maxima.indices
    result[nmask] = data[nmask]
    return result


@th.inference_mode()
def predict(
    model: ModelBase,  # should return single token
    window: int,  # model input window length
    count: int,  # output length
    temperature: float = 1.0,
    seed: np.ndarray | None = None,
    cleanup_model: ModelBase | None = None,  # should return token sequence
    cleanup_window: int = 1,  # cleanup input window length
    cleanup_runup: int = 1,  # number of steps to generate before running cleanup
    cleanup_threshold: float = 0.1,  # minimum probability difference required to replace a token
    cleanup_all: bool = False  # whether cleanup can modify its entire input window (vs only the runup)
) -> np.ndarray:
    if seed is None:
        seed = np.array([0])

    device = next(model.parameters()).device
    data = th.LongTensor(seed).to(device)
    mask = th.ones(model.data_fmt.EVENT_COUNT, dtype=bool).to(device)  # next token mask
    runup = 0

    # WARN: very slow code with exactly zero caching and optimization
    for i in (tqdm_it := tqdm(range(count), leave=False)):
        if len(data) > 0:
            mask = th.BoolTensor(model.data_fmt.next_mask(data[-1])).to(device)

        logits = model(data[-window:])
        logits[~mask] = -th.inf

        if math.isclose(temperature, 0.0, abs_tol=1e-3):
            result = th.argmax(logits, keepdim=True)
        else:
            result = th.multinomial(th.nn.functional.softmax(logits / temperature, dim=-1), 1)

        data = th.cat((data, result))

        if cleanup_model is not None:
            runup += 1
            if (len(data) >= cleanup_window) and ((runup >= cleanup_runup) or (i + 1 >= count)):
                modsize = min(len(data) - len(seed), cleanup_window if cleanup_all else runup)
                cdata = cleanup(cleanup_model, data[-cleanup_window:], cleanup_threshold)[-modsize:]
                tqdm_it.write(f"Cleanup: {int(th.sum(data[-modsize:] != cdata))}/{modsize}")
                data[-modsize:] = cdata
                runup = 0

    return np.array(data.cpu())

