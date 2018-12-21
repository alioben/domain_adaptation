# Residual Parameter Transfer for Deep Domain Adaptation
Implementation of the CVPR 2017 paper published by Artem Rozantsev et al, *Residual Parameter Transfer for Deep Domain Adaptation*

# Train on MNIST->SVNH
1. Download the "MNIST" and "USPS" datasets from [here](https://www.dropbox.com/s/wanwuhh1cwf6krs/mnist_svhn.zip?dl=0)
2. Donlowad the indices for MNIST/SVNH from [here](https://drive.google.com/drive/folders/1N5VQ8ry-tHr53cXjeBzlPifCcMN1VCsb?usp=sharing)
2. Extract and place both training and indices files in `mnist_svnh` folder under `dataset` as follows:
```
  domain_adaptation/dataset/mnist_svnh/mnist_60k_ids.npy
  domain_adaptation/dataset/mnist_svnh/svnh_73k_ids.npy
  domain_adaptation/dataset/mnist_svnh/train_mnist_images.npy
  domain_adaptation/dataset/mnist_svnh/train_mnist_labels.npy
  domain_adaptation/dataset/mnist_svnh/train_svnh_images.npy
  domain_adaptation/dataset/mnist_svnh/train_svnh_labels.npy
```
3. Tain on svnh/mnist domain adaptation: `python train.py`

**Note:** This implementation works for MNIST --> SVHN (source --> target). I have not tried yet for MNIST --> USPS.
