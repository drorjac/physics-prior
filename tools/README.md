# tools

Scripts and files that sit outside the package. All of them assume it is
installed (`pip install -e '.[all]'`), except the two device benchmarks.

## Pipeline and checks

| Script | What it does |
|---|---|
| `regenerate.sh` | the full pipeline: run every problem, run the neglected-terms study, write figures, regenerate the docs and README tables, execute the notebooks, then run the tests |
| `validate_palette.py` | run the figure-palette validator without installing the package (it is also `python -m physprior.viz.palette`) |
| `audit_pinn.py` | where the `pinn` arm wins, loses and ties against the best non-oracle arm, with a gap smaller than the seed spread reported as a tie (the tables in `docs/plans/PLAN.md` §2) |

## The pinned environment

`requirements.lock` pins the environment that produced `results/`:

```bash
pip install -e '.[all]' -c tools/requirements.lock
```

## Device benchmarks

`device_speed.py` and `device_precision.py` are standalone, kept out of the package because they exist to be run
against a **different torch** from the one the project pins.

| file | question |
|---|---|
| `device_speed.py` | how much faster is the 2-D field residual on a GPU? |
| `device_precision.py` | what does the float32 a GPU forces actually cost? |

They import nothing from `physprior`; the PINN inner loop is replicated so
they can run in a bare venv.

### Running them where the GPU actually works

This machine is an **Apple M2, 10 GPU cores, Metal 3**; the hardware is
there. The project's own torch cannot reach it:

```
$ python -c "import torch; torch.ones(3, device='mps')"
RuntimeError: The MPS backend is supported on MacOS 14.0+
```

torch 2.11 requires macOS 14; this machine runs 13.4. So `mps.is_built()` is
True (the code is compiled in) while `mps.is_available()` is False (the OS is
too old). **That is an OS constraint, not a missing GPU and not missing
code.** PyTorch raised the floor at 2.9, so any earlier build reaches it:

```bash
python3.13 -m venv /tmp/gpuenv
/tmp/gpuenv/bin/pip install 'torch==2.8.0' numpy
/tmp/gpuenv/bin/python tools/device_speed.py
/tmp/gpuenv/bin/python tools/device_precision.py
```

Verified on this machine: `torch 2.8.0`, `mps built True | avail True`,
`tensor([1., 1., 1.], device='mps:0')`.

The two options for using the GPU from the project itself are therefore
**upgrade macOS to 14+**, or **pin torch <= 2.8**. Neither is done here: the
committed results are CPU/float64 and stay that way.
