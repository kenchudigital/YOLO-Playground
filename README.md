# YOLO-Playground
YOLO playground

## Fall Detection

### Get Started

First, you need to download the dataset from Kaggle [LINK](!https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset?resource=download)

After that put the folder to `datasets/fall_datasets`

File Structure:

Fall_Detection
    - datasets/fall_datasets/..
    - detect.py
    - train.py
    - test_image.jpg
    - data.yaml

Finally, you can train the model:

```

python train.py

```

After that, it will output `/runs` and we get the best model we trained: `runs/detects/elderly_fall_detection_cpu.weights/best.pt` (may be different in your path) so you may need to change the path in `detect.py`

Finally, use this trained model to detect falling:

```bash

python detect.py

```

![img](/detected_output.jpg)

Success !


# Common Error

- remove `Ultralytics/settings.json' for reset the path if you change the path