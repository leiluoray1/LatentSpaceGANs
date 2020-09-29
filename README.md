## LSGANs: Towards Fine-grained Control on Latent Space for Unpaired Image-to-Image Translation

### edge to shoes Translation

![](results/edges2shoes.png)

### Cat to dog Translation

![](results/cat2dog_final.png)

### Dog to cat interpolation

![](results/dog2cat_transfer.png)

### Cat to dog interpolation

![](results/cat2dog_transfer.png)


#### Testing 
```
python test.py
```
 
#### Training
```
python train.py --config configs/edges2handbags.yaml
```

## Reference
Some code are adapted from online.
* [MUNIT](https://github.com/NVlabs/MUNIT)


