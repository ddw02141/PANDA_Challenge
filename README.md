# PANDA_Challenge

25th place solution for [Prostate cANcer graDe Assessment (PANDA) Challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment)

### About

Our code is based on [Qishen Ha's code](https://www.kaggle.com/haqishen/train-efficientnet-b0-w-36-tiles-256-lb0-87). Difference in our train code is

* batch_size : 2 => 8
* model : efficientnet-b0 => efficientnet-b1

We made a change in our inference code - TTA(Test Time Augmentation). You can see our inference code in below link.

* [Inference code](https://www.kaggle.com/kyunghoonhur/efficientnet-b0-b1-1-1-ensemble-0-92664/)
* [Qishen Ha's pre-trained weight file](https://www.kaggle.com/haqishen/panda-public-models)

### How to run

We run our code with 4 V100 GPU. You can change GPU number in those two lines.

```
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
```
```
model = nn.DataParallel(model, device_ids = [0,1,2,3])
```

Also, you can change fold number from 0 to 4 to select different fold as validation set. 

```
fold = 0
```

At final submission, we use 2 weight file generated from fold=0 and fold=1.


