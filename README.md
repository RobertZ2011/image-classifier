# Image Classifier

Utility script for trainer, evaluating, and running inference on image classifier models.

## Usage

### Finetune
```
usage: image-classifier.py tune [-h] [-e EPOCHS] [-m MODEL] [-d DATASET] [-s SPLIT] [-p PROCESSOR] out_dir

positional arguments:
  out_dir               Output directory

options:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -m MODEL, --model MODEL
                        Model name/path
  -d DATASET, --dataset DATASET
                        Dataset for evaluation
  -s SPLIT, --split SPLIT
  -p PROCESSOR, --processor PROCESSOR
                        Device to use
```

### Evaluate
```
usage: image-classifier.py eval [-h] [-m MODEL] [-d DATASET] [-s SPLIT] [-p PROCESSOR]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model name/path
  -d DATASET, --dataset DATASET
                        Dataset for evaluation
  -s SPLIT, --split SPLIT
  -p PROCESSOR, --processor PROCESSOR
                        Device to use
```

### Run Inference
```
usage: image-classifier.py run [-h] [-m MODEL] [-p PROCESSOR] images [images ...]

positional arguments:
  images                Path to the images to classify

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model name/path
  -p PROCESSOR, --processor PROCESSOR
                        Device to use
```
