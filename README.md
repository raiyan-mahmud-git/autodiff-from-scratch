# TinyDL: autodiff from scratch

TinyDL is a small NumPy-based neural-network library that implements reverse-mode automatic differentiation and the pieces needed to train a handwritten-digit classifier. It is an educational portfolio project with vectorized tensors, gradient checks, optimizer experiments, and a browser demo.

## Live demo

**Live demo:** [Open the TinyDL digit demo](https://raiyan-mahmud-git.github.io/autodiff-from-scratch/). To run it locally, open `web/index.html`; keep `app.js` and `forward.js` beside it.

## Credit

This project is inspired by Andrej Karpathy's [micrograd tutorial and repository](https://github.com/karpathy/micrograd). My contributions are vectorized tensors, broadcasting support, PyTorch comparison tests, experiments, and the browser demo.

## Results

| Run | Seeds | Test set | Test accuracy (mean ± sample standard deviation) |
|---|---|---:|---:|
| 784-128-64-10 MLP, Adam, validation-selected learning rate and checkpoint | 0, 1, 2, 3, 4 | 10000 images | 0.9734 ± 0.0023 |

The result above is copied from `python experiments/train_mnist.py`. The run split the original training set into 55000 training and 5000 validation images; the test set was evaluated after the five seed-specific checkpoints had been selected. These test results describe the non-augmented five-seed training run. The browser demo uses a separate affine-augmented seed-zero checkpoint with digit-9 calibration selected on validation data; that demo checkpoint has no reported test-set score.

## How correctness is checked

The finite-difference checker uses central differences with `eps = 1e-5` and reports maximum relative error. Tests use a threshold below `1e-6` for simple operations and below `1e-5` for compound expressions, including broadcasting and repeated tensor use.

Optional PyTorch tests compare forward outputs, cross-entropy, every parameter gradient, and 200 training steps for SGD and Adam at `1e-8` tolerance. They skip with a clear message unless the optional `compare` extra is installed. The JavaScript forward pass is checked against the exported Python vectors with a `1e-4` tolerance.

## Run and reproduce

Requires Python 3.11 or 3.12. NumPy is the only runtime dependency. MNIST is downloaded from the CVDF mirror and checked against the published MD5 checksums before parsing.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check src tests experiments
```

Install PyTorch only when you want to run the reference comparisons:

```powershell
python -m pip install -e ".[compare]"
python -m pytest tests/test_vs_pytorch.py -q
```

Reproduce the baseline five-seed training output and loss plot:

```powershell
python experiments/train_mnist.py
```

Run the experiments and prepare the robust demo model:

```powershell
python experiments/optimizer_comparison.py
python experiments/dropout_ablation.py
python experiments/init_comparison.py
python experiments/handwriting_robustness.py
python experiments/calibrate_digit9.py
python -m tinydl.export
```

The optimizer, dropout, initialization, and affine-robustness experiments use seeds `0, 1, 2, 3, 4` by default, print measured tables, and write plots under `artifacts/`. The digit-nine calibration uses only the seed-zero checkpoint and the validation split. The export command writes `web/model.json`, `web/test_vectors.json`, and `web/index.html`. To check the web outputs after export:

```powershell
node web/verify_forward.js web/model.json web/test_vectors.json
node web/verify_segmentation.js
```

## Limitations

Training is CPU-only and intentionally small-scale; the library has no convolution layer. The digit model is an MLP, so performance on hand-drawn inputs can differ from MNIST validation and test images. Multi-digit recognition splits digits at visible vertical gaps; touching digits are not reliably segmented. The web model's digit-9 calibration improves recall at the cost of some precision.

## What I learned

I learned how reverse-mode backpropagation accumulates gradients through a computation graph, why broadcasted dimensions must sum their gradient contributions, and how numerical gradient checks catch mistakes in local derivative rules. I also learned to separate training, validation, and test decisions, compare implementations against an independent framework, and check the limits of a model before making claims about it.
