# 🕹️ HyperPlay

🚧 Work in Progress 🚧

HyperPlay is an interactive GUI for exploring how hyperparameters affect a
simple neural network learning on 2D toy datasets. Adjust the settings and
watch the decision boundary update live.

## Quick Start

```bash
# clone this repo
git clone https://github.com/nick-ob/hyperplay
cd hyperplay

# install requirements
pip install -r requirements.txt

# run the app
python main.py
```

## What This Project Includes

- Live decision boundary visualisation for 2D datasets
- Custom fully connected neural network (NumPy only)
- Background training thread for smooth UI updates
- Dataset loading from CSV files in `data/`
- Extensible GUI layout for hyperparameter controls

## Datasets

Datasets are stored in `data/` as CSV files. Each file contains columns for the
input features and a `y` label column.

Included datasets:

- `xor.csv`
- `spiral.csv`
- `moons.csv`
- `circles.csv`
- `blobs.csv`

## Project Structure

```text
hyperplay/
|-- main.py
|-- requirements.txt
|-- data/
|   |-- xor.csv
|   |-- spiral.csv
|   |-- moons.csv
|   |-- circles.csv
|   |-- blobs.csv
|-- src/
|   |-- gui.py                # GUI and training loop
|   |-- model.py              # Network class
|   |-- layer.py              # layer implementation
|   |-- activations.py        # ReLU / Softmax
|   |-- loss.py               # CCE loss + accuracy metric
|   |-- loading.py            # loads datasets from CSV
|   |-- types.py              # shared types for UI
```

## Tech Stack

- **Python**
- **NumPy** - math
- **CustomTkinter** - GUI
- **Matplotlib, Seaborn** - visualisations

## Status

The core training + visualisation loop is implemented. The next focus is the
hyperparameter controls (dataset picker, architecture editor, sliders) and UI
polish.

## License

MIT — See [LICENSE](LICENSE)
