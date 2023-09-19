This repository was an experiment in symbolic music generation via deep learning.
Two models &mdash; an LSTM and a Transformer &mdash; of similar sizes (~4M parameters)
were trained on a fairly modest dataset of Final Fantasy soundtracks arranged for piano,
after which their quality was empirically evaluated.
Experiment details and results can be found in the [paper](paper.pdf).

Models were trained as follows:
```sh
# LSTM
python lstm.py lstm-64-512-2 --sequence --batch-size=8,256 --epoch-size=0.02,0.35 --lr=0.0005 --lr-decay=0.8 --lr-limit=0.00005

# Transformer
python transformer.py transformer-256-16-8 --sequence --batch-size=8,64 --epoch-size=0.02,0.35 --lr=0.001 --lr-decay=1.0

# FNet (unused)
python fnet.py fnet-384-8 --batch-size=8,128 --epoch-size=0.05,0.35 --lr=0.01 --lr-decay=1.0
```

Code examples for actual music generation are given at the end of [utils.py](utils.py).
The dataset used is in [input/ff/](input/ff/). Generated samples are in [output/](output/).
