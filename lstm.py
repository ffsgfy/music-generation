import time
import math

import numpy as np
import torch as th
from tqdm import tqdm

import dataset


class Model(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = th.nn.Embedding(dataset.EventFormat.EVENT_COUNT, 64)
        self.lstm = th.nn.LSTM(input_size=64, hidden_size=384, num_layers=2, dropout=0.5)
        self.decoder = th.nn.Sequential(
            th.nn.Dropout(0.5),
            th.nn.Linear(in_features=384, out_features=512),
            th.nn.LayerNorm(512),
            th.nn.ReLU(),
            th.nn.Dropout(0.5),
            th.nn.Linear(in_features=512, out_features=256),
            th.nn.LayerNorm(256),
            th.nn.ReLU(),
            th.nn.Dropout(0.5),
            th.nn.Linear(in_features=256, out_features=dataset.EventFormat.EVENT_COUNT),
        )

        self.data_fmt = dataset.EventFormat()

    def forward(
        self, x: th.Tensor, hx: tuple[th.Tensor, th.Tensor] | None = None, sequence: bool = False
    ) -> th.Tensor:
        x = self.lstm(self.encoder(x), hx)[0]
        x = self.decoder(x if sequence else x[-1])
        return th.nn.functional.log_softmax(x, dim=-1)

    def predict(
        self, data: np.ndarray, length: int, temperature: float = 1.0, masked: bool = False
    ) -> np.ndarray:
        device = next(self.parameters()).device
        x = th.LongTensor(data).to(device)
        hx = None
        mask = th.ones(self.data_fmt.EVENT_COUNT, dtype=bool).to(device)  # next token mask
        result = []

        with th.no_grad():
            for i in range(length):
                if masked:
                    mask = th.BoolTensor(self.data_fmt.next_mask(x[-1])).to(device)

                x, hx = self.lstm(self.encoder(x), hx)
                x = self.decoder(x[-1])
                x[~mask] = -th.inf

                if math.isclose(temperature, 0.0, abs_tol=1e-9):
                    x = th.argmax(x, dim=-1, keepdim=True)
                else:
                    x = th.nn.functional.softmax(x / temperature, dim=-1)
                    x = th.multinomial(x, 1)

                result.append(int(x))

        return np.array(result)


if __name__ == "__main__":
    th.manual_seed(42)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    window = 256
    batch_size = 64

    model_ts = int(time.time())
    model = Model().to(device)
    loss_fn = th.nn.NLLLoss()
    optimizer = th.optim.Adamax(model.parameters(), lr=0.01)

    train_ds = dataset.load_folder("input/ff/train/", window=window, fmt=model.data_fmt)
    train_dl = th.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds = dataset.load_folder("input/ff/test/", window=window, fmt=model.data_fmt)
    test_dl = th.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(30):
        model_path = f"models/lstm-{model_ts}-{epoch}.pth"
        print("\n", model_path)

        model.train()
        loss_sum = 0.0
        for i, (inputs, targets) in enumerate(tqdm_it := tqdm(train_dl, desc="Training")):
            inputs = inputs.transpose(0, 1).to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss)
            tqdm_it.set_description(f"Training ({loss_sum / (i + 1):.05f})")

        th.save(model.state_dict(), model_path)

        model.eval()
        loss_sum = 0.0
        with th.no_grad():
            for i, (inputs, targets) in enumerate(tqdm_it := tqdm(test_dl, desc="Testing")):
                inputs = inputs.transpose(0, 1).to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss_sum += float(loss)
                tqdm_it.set_description(f"Testing ({loss_sum / (i + 1):.05f})")


