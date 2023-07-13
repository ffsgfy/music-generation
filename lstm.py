import time

import numpy as np
import torch as th
from tqdm import tqdm

import dataset


class Model(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = th.nn.LSTM(input_size=88, hidden_size=512, num_layers=2)
        self.decoder = th.nn.Sequential(
            th.nn.Linear(in_features=512, out_features=512),
            th.nn.ReLU(),
            th.nn.Dropout(p=0.3),
            th.nn.Linear(in_features=512, out_features=256),
            th.nn.ReLU(),
            th.nn.Dropout(p=0.2),
            th.nn.Linear(in_features=256, out_features=128),
            th.nn.ReLU(),
            th.nn.Dropout(p=0.1),
            th.nn.Linear(in_features=128, out_features=88),
            th.nn.Sigmoid()
        )

    def forward(self, x: th.Tensor, hx: tuple[th.Tensor, th.Tensor] | None = None) -> th.Tensor:
        return self.decoder(self.lstm(x, hx)[1][0][-1])

    def predict(self, roll: np.ndarray, length: int, threshold: float = 0.3) -> np.ndarray:
        device = next(self.parameters()).device
        x = th.Tensor(roll).to(device)
        hx = None
        result = []

        with th.no_grad():
            for i in range(length):
                _, hx = self.lstm(x, hx)
                output = self.decoder(hx[0][-1])
                mask = output > threshold
                output[mask] = 1.0
                output[~mask] = 0.0
                x = output.unsqueeze(0)
                result.append(np.array(output))

        return np.stack(result)


if __name__ == "__main__":
    th.manual_seed(42)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    window = 48 * dataset.RESOLUTION
    batch_size = 64

    model_ts = int(time.time())
    model = Model().to(device)
    model.train()
    loss_fn = th.nn.BCELoss()
    loss_ema = 0.0
    optimizer = th.optim.Adamax(model.parameters(), lr=0.01)

    train_ds = dataset.load_folder("input/train/", window=window)
    train_dl = th.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds = dataset.load_folder("input/test/", window=window)
    test_dl = th.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(10):
        tqdm_it = tqdm(train_dl, desc="Training")
        for inputs, targets in tqdm_it:
            inputs = inputs.transpose(0, 1).to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            loss_ema = loss_ema * 0.5 + float(loss) * 0.5
            tqdm_it.set_description(f"Training ({loss_ema:.05f})")

        loss_sum = 0.0
        with th.no_grad():
            for inputs, targets in tqdm(test_dl, desc="Testing"):
                inputs = inputs.transpose(0, 1).to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss_sum += float(loss_fn(outputs, targets))

        model_path = f"models/lstm-{model_ts}-{epoch}.pth"
        th.save(model.state_dict(), model_path)
        print(f"Epoch: {epoch}, test loss: {loss_sum / len(test_dl):.09f}, path: {model_path}")

