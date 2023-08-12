import fire
import torch as th

import utils


class Model(utils.ModelBase):
    def __init__(self):
        super().__init__()

        d_input = 64
        d_model = 512
        self.encoder = th.nn.Embedding(self.data_fmt.EVENT_COUNT, d_input)
        self.lstm = th.nn.LSTM(
            input_size=d_input, hidden_size=d_model, num_layers=2, dropout=0.5, batch_first=True
        )
        self.decoder = th.nn.Sequential(
            th.nn.Dropout(0.5),
            th.nn.Linear(in_features=d_model, out_features=d_model * 2),
            th.nn.LayerNorm(d_model * 2),
            th.nn.ReLU(),
            th.nn.Dropout(0.5),
            th.nn.Linear(in_features=d_model * 2, out_features=d_model),
            th.nn.LayerNorm(d_model),
            th.nn.ReLU(),
            th.nn.Dropout(0.5),
            th.nn.Linear(in_features=d_model, out_features=self.data_fmt.EVENT_COUNT),
        )

    def forward(
        self, x: th.Tensor, hx: tuple[th.Tensor, th.Tensor] | None = None, sequence: bool = False
    ) -> th.Tensor:
        x = self.lstm(self.encoder(x), hx)[0]
        return self.decoder(x if sequence else x[..., -1, :])


def train(
    model_id: str,
    state_path: str | None = None,
    epoch_size: int | float | tuple[int | float, int | float] = 1.0,
    epoch_start: int | None = None,
    epoch_limit: int = 50,
    window: int = 256,
    batch_size: int | tuple[int, int] = 64,
    lr: float = 0.01,
    lr_decay: float = 0.9,
    lr_limit: float = 0.0,
    sequence: bool = False
):
    def batch_fn(batch: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        nonlocal sequence

        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        if not sequence:
            targets = targets[:, -1]

        return (inputs, targets)

    def loss_fn(model: Model, batch: tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        nonlocal sequence

        inputs, targets = batch
        outputs = th.nn.functional.log_softmax(model(inputs, sequence=sequence), dim=-1)
        if sequence:
            outputs = outputs.transpose(-1, -2)

        return th.nn.functional.nll_loss(outputs, targets)

    utils.train(
        Model(), model_id, batch_fn, loss_fn,
        state_path, epoch_size, epoch_start, epoch_limit, window + 1, batch_size, lr, lr_decay, lr_limit
    )


if __name__ == "__main__":
    fire.Fire(train)

