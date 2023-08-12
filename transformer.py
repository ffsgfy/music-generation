import math

import fire
import torch as th

import utils


class Attention(th.nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_pos: int):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_pos = n_pos

        self.pe = th.nn.Parameter(th.randn((n_pos, self.d_head)))  # relative positional
        self.qe = th.nn.Linear(d_model, d_model)  # query
        self.ke = th.nn.Linear(d_model, d_model)  # key
        self.ve = th.nn.Linear(d_model, d_model)  # value
        self.oe = th.nn.Linear(d_model, d_model)  # output

        # Initialize projection weights
        for linear in (self.qe, self.ke, self.ve, self.oe):
            th.nn.init.xavier_normal_(linear.weight)
            th.nn.init.zeros_(linear.bias)

    def forward(self, x: th.Tensor, attn_residual: th.Tensor | None = None) -> th.Tensor:
        seq_len = x.size(-2)

        # (*, seq_len, d_model)
        Q = self.qe(x)
        K = self.ke(x)
        V = self.ve(x)

        # (*, n_heads, seq_len, d_head)
        Q = Q.unflatten(-1, (self.n_heads, self.d_head)).transpose(-2, -3)
        K = K.unflatten(-1, (self.n_heads, self.d_head)).transpose(-2, -3)
        V = V.unflatten(-1, (self.n_heads, self.d_head)).transpose(-2, -3)

        # (seq_len, d_head)
        if seq_len <= self.n_pos:
            pe = self.pe[self.n_pos - seq_len:]
        else:
            pe = th.cat((self.pe[0].expand((seq_len - self.n_pos, -1)), self.pe))

        P = th.matmul(Q, pe.transpose(-1, -2))  # (*, n_heads, seq_len, seq_len)
        P = th.nn.functional.pad(P, (1, 0))  # (*, n_heads, seq_len, seq_len + 1)
        P = P.reshape(P.shape[:-2] + (P.size(-1), P.size(-2)))  # (*, n_heads, seq_len + 1, seq_len)
        P = P[..., 1:, :]  # (*, n_heads, seq_len, seq_len)

        # (*, n_heads, seq_len, seq_len)
        attn = (th.matmul(Q, K.transpose(-1, -2)) + P) / math.sqrt(self.d_head)
        if attn_residual is not None:
            attn += attn_residual
        attn_residual = attn
        attn_mask = ~th.tril(th.ones((seq_len, seq_len), dtype=bool, device=attn.device))
        attn = attn.masked_fill(attn_mask, -th.inf)
        attn = th.nn.functional.softmax(attn, dim=-1)

        out = th.matmul(attn, V)  # (*, n_heads, seq_len, d_head)
        out = out.transpose(-2, -3).flatten(-2, -1)  # (*, seq_len, d_model)
        out = self.oe(out)  # (*, seq_len, d_model)

        return (out, attn_residual)


class Layer(th.nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_pos: int, ff_exp: int = 2, dropout: float = 0.0):
        super().__init__()

        self.attn = Attention(d_model, n_heads, n_pos)
        self.ff = th.nn.Sequential(
            th.nn.Linear(d_model, d_model * ff_exp),
            th.nn.ReLU(),
            th.nn.Linear(d_model * ff_exp, d_model),
        )
        self.norm1 = th.nn.LayerNorm(d_model)
        self.norm2 = th.nn.LayerNorm(d_model)
        self.dropout = th.nn.Dropout(dropout)

    def forward(self, x: th.Tensor, attn_residual: th.Tensor | None = None) -> tuple[th.Tensor, th.Tensor]:
        x_attn, attn_residual = self.attn(x, attn_residual)
        x = self.norm1(self.dropout(x_attn) + x)
        x = self.norm2(self.dropout(self.ff(x)) + x)
        return (x, attn_residual)


class Model(utils.ModelBase):
    def __init__(self):
        super().__init__()

        d_model = 256
        self.encoder = th.nn.Embedding(self.data_fmt.EVENT_COUNT, d_model)
        self.layers = th.nn.ModuleList(
            Layer(d_model=d_model, n_heads=16, n_pos=240, ff_exp=2, dropout=0.5) for _ in range(8)
        )
        self.decoder = th.nn.Linear(in_features=d_model, out_features=self.data_fmt.EVENT_COUNT)

    def forward(self, x: th.Tensor, sequence: bool = False) -> th.Tensor:
        x = self.encoder(x)
        attn_residual = None

        for layer in self.layers:
            x, attn_residual = layer(x, attn_residual)

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

