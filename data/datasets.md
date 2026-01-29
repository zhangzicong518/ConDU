## Data preparation

Datasets list:
- [Caltech101](#caltech101)
- [OxfordPets](#oxfordpets)
- [StanfordCars](#stanfordcars)
- [Flowers102](#flowers102)
- [Food101](#food101)
- [FGVCAircraft](#fgvcaircraft)
- [SUN397](#sun397)
- [DTD](#dtd)
- [EuroSAT](#eurosat)
- CIFAR100
- MNIST

The instructions to prepare each dataset are detailed below. If you have problems with Caltech101, you can refer to [issue#6](https://github.com/JiazuoYu/MoE-Adapters4CL/issues/6). More details can refer to [datasets.md](mtil%2Fdatasets.md) of [ZSCL](https://github.com/Thunderbeee/ZSCL).

The file structure looks like

~~~
data
├── caltech101
├── cifar-100-python
├── dtd
├── eurosat
├── fgvc-aircraft-2013b
├── flowers-102
├── food-101
├── MNIST
├── oxford-iiit-pet
├── stanford_cars
├── SUN397
~~~


### Caltech101
- Create a folder named `caltech101/` under `data`.
- Download `101_ObjectCategories.tar.gz` from http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract the file under `data/caltech101`.

### OxfordPets
- Create a folder named `oxford-iiit-pet/` under `data`.
- Download the images from https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz.
- Download the annotations from https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz.

### StanfordCars
- Create a folder named `stanford_cars/` under `data`.
- Download the train images http://ai.stanford.edu/~jkrause/car196/cars_train.tgz.
- Download the test images http://ai.stanford.edu/~jkrause/car196/cars_test.tgz.
- Download the train labels https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz.
- Download the test labels http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat.

### Flowers102
- Create a folder named `flowers-102/` under `data`.
- Download the images and labels from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz and https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat respectively.

### Food101
- Download the dataset from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ and extract the file `food-101.tar.gz` under `data`, resulting in a folder named `data/food-101/`.


### FGVCAircraft
- Download the data from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Move `data/` to `data` and rename the folder to `fgvc-aircraft-2013b/`.

### SUN397
- Create a folder named  `SUN397/` under `data`.
- Download the images http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz.
- Download the partitions https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip.
- Extract these files under `data/SUN397/`.

### DTD
- Download the dataset from https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz and extract it to `data`. This should lead to `data/dtd/`.

### EuroSAT
- Create a folder named `eurosat/` under `data`.
- Download the dataset from http://madm.dfki.de/files/sentinel/EuroSAT.zip and extract it to `data/eurosat/`.