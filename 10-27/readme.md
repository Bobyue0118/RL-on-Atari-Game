# 10-27 night

## DQN_basic.py

A basic model which supports tensorboard during training.

#### training:
```
python DQN_basic.py  --name test_name
```

#### tensorboard:
```
tensorboard --logdir=checkpoints/
```


## test_model.py

Load trained model, and test it  in the game

####  test
```
python test_model.py --pretrained-model  /path/to/the/model
```
