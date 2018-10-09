# cell_segmentation
Segmenting cells from lungmap images.

## Purpose
The purpose of this repository is to start with the assumption that the immunofluorescence confocal 
images have adequately had instance segmentation applied. Once at this point, I can assume that I 
have a contour, label, and probability of anatomy. At this point, my job is to further provide instance
segments of cells within these larger anatomical entities and ideally tie these to an ontology which
allows me to tie the color of the cell to the type of cell segmented.

## Usage

To run the entire pipeline:

1. Run `data_generator.py`: This will produce the training and test data in a `model_data` directory.
1. Run `xception_transfer.py`: This will train the Xception pipeline, and can take quite a while.
1. Run `test.py`: This will evaluate the test image for accuracy, saving the results in `results.csv` in the `model_data` directory.

