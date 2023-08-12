import fire
import torch as th

import dataset
import utils


class Layer(th.nn.Module):
    def __init__(self, d_model: int, ff_exp: int = 2, dropout: float = 0.0):
        super().__init__()

        self.ff = th.nn.Sequential(
            th.nn.Linear(d_model, d_model * ff_exp),
            th.nn.ReLU(),
            th.nn.Dropout(dropout),
            th.nn.Linear(d_model * ff_exp, d_model),
            th.nn.Dropout(dropout),
        )
        self.norm1 = th.nn.LayerNorm(d_model)
        self.norm2 = th.nn.LayerNorm(d_model)

    def forward(self, x: th.Tensor, residual: th.Tensor | None = None) -> tuple[th.Tensor, th.Tensor]:
        x_residual = th.fft.fft2(x).real
        if residual is not None:
            x_residual = (x_residual + residual) * 0.5

        x = self.norm1(x_residual + x)
        x = self.norm2(self.ff(x) + x)

        return (x, x_residual)


class Model(utils.ModelBase):
    def __init__(self):
        super().__init__()

        d_model = 384
        self.encoder = th.nn.Embedding(self.data_fmt.EVENT_COUNT, d_model)
        self.layers = th.nn.ModuleList(Layer(d_model=d_model, ff_exp=2, dropout=0.0) for _ in range(8))
        self.decoder = th.nn.Linear(in_features=d_model, out_features=self.data_fmt.EVENT_COUNT)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.encoder(x)
        x_residual = None

        for layer in self.layers:
            x, x_residual = layer(x, x_residual)

        return self.decoder(x)


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
    mask_prob: float | str = 0.2
):
    if isinstance(mask_prob, str):
        mask_prob = th.load(mask_prob)
    else:
        event_count = dataset.EventFormat.EVENT_COUNT
        mask_prob_scalar = mask_prob
        mask_prob = th.empty((event_count, event_count))
        mask_prob.fill_(mask_prob_scalar / (event_count - 1))
        mask_prob.fill_diagonal_(1.0 - mask_prob_scalar)

    def batch_fn(batch: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        nonlocal mask_prob

        mask_prob = mask_prob.to(batch.device)
        inputs = th.multinomial(mask_prob[batch].flatten(0, -2), 1).reshape(batch.shape)

        return (inputs, batch)

    def loss_fn(model: Model, batch: tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        inputs, targets = batch
        outputs = th.nn.functional.log_softmax(model(inputs), dim=-1)
        mask = inputs != targets

        return th.nn.functional.nll_loss(outputs[mask], targets[mask])

    utils.train(
        Model(), model_id, batch_fn, loss_fn,
        state_path, epoch_size, epoch_start, epoch_limit, window, batch_size, lr, lr_decay, lr_limit
    )


if __name__ == "__main__":
    fire.Fire(train)

