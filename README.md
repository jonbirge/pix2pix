Pix2Pix
======================

[Pix2Pix](https://arxiv.org/abs/1609.04802) is an example of `image-to-image` translation GAN.

### Python Dependencies

- torch
- torchvision
- progress
- tqdm

### Usage

#### Datasets
This project includes three datasets:

- **Cityscapes**
- **Facades**
- **Maps**

Each of them can be downloaded automatically by following:
```python
from dataset import Cityscapes, Facades, Maps

dataset = Facades(root='.',
                  transform=None, # if transform is None, then it returns PIL.Image
                  download=True,
                  mode='train',
                  direction='B2A')
```

#### Transforms
For simple conversion of imageA and imageB from `PIL.Image` to `torch.Tensor` use `dataset.transforms`:
```python
from dataset import transforms as T
# it works almost like `torchvision.transforms`
transforms = T.Compose([T.Resize(size=(..., ...)), # (width, height)
                        T.CenterCrop(size=(..., ...), p=...), # (width, height); probability of crop/else resize
                        T.Rotate(p=...), # probability of rotation on 90'
                        T.HorizontalFlip(p=...), # probability of horizontal flip
                        T.VerticalFlip(p=...), # probability of vertical flip
                        T.ToTensor(), # to convert PIL.Image to torch.Tensor
                        T.Normalize(mean=[..., ..., ...],
                                     std=[..., ..., ...])])
```
As input, `transforms` take one/or two arguments (`imageA`/or `imageA` and `imageB`):
```python
imgA_transformed = transforms(imgA)
# or
imgA_transformed, imgB_transformed = transforms(imgA, imgB)
```
There are also other `transforms`:
```python
T.ToImage() # to convert torch.Tensor to PIL.Image 
T.DeNormalize(mean=[..., ..., ...],
               std=[..., ..., ...])])
```

## License

This project is licensed under MIT.

## Links

* [Image-to-Image Translation with Conditional Adversarial Networks (arXiv)](https://arxiv.org/pdf/1611.07004.pdf)
