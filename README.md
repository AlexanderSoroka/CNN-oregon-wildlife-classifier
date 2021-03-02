# Oregon wildlife classification example using CNN on tf 2.x + keras

The goal of that lab is to create CNN that solves Oregon wildlife Classification task

Pre-requisites:
1. TensorFlow 2.x environment

Steps to reproduce results:
1. Clone the repository:
```
git clone git@github.com:AlexanderSoroka/CNN-oregon-wildlife-classifier.git
```
2. Download [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife) from kaggle to archive.zip
- unpack dataset `unzip archive.zip`
- change current directory to the folder with unpacked dataset

3. Generate TFRecords with build_image_data.py script:

```
python build_image_data.py --input <dataset root path> --output <tf output path>
```

Validate that total size of generated tfrecord files is close ot original dataset size

4. Run train.py to train pre-defined CNN:
```
python train.py --train '<dataset root path>/train*'
```

5. Modify model and have fun
