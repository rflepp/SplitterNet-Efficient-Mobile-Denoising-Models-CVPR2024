## SplitterNet and more Mobile Image Denoising Deep Learning Models

<br/>

<img src="https://github.com/rflepp/SplitterNet-Efficient-Mobile-Denoising-Models-CVPR2024/blob/main/images/SplitterNet_arch.png"/>

<br/>

#### 1. Overview
This is the official repository for the CVPR2024 paper "Real-World Mobile Image Denoising Dataset with Efficient Baselines". It includes the presented SplitterNet as well as other state of the art efficient denoising networks, optimized for the mobile usage with .TFLite. A simple U-Net, the winning models of the [MAI2021](https://arxiv.org/pdf/2105.08629v1.pdf) challenge by NOAHTCV and Megvii, the newly proposed SplitterNet and MoDeNet as well as some other implementations are included.

The presented MIDD dataset is not yet publicly available. Updates will be found on [Project Webpage](https://people.ee.ethz.ch/~ihnatova/midd.html#)

<br/>

#### 2. Prerequisites

- Python: scipy, Nmpy, imageio
- [TensorFlow 2.X](https://www.tensorflow.org/install/) + [CUDA](https://developer.nvidia.com/cuda-toolkit)
- GPU cluster using Slurm

<br/>

#### 3. First steps

- Add the wanted dataset. If the dataset is not cropped into patches yet, create those using the ```data_preprocessing > cropping_parallel.py``` file. Please make sure to see the data loading function to set your paths and the needed file pairing algorithm. For MIDD you can have the configuration of one noisy to one denoised or 20 noisy to the same denoised image.
- Add the wanted testset. It needs to have a form with two subfolders ```/denoised/``` and ```/original/```
- Add the ABSPATH variable in the run_evaluation.sh and run_training.sh files to the wanted folder.
- Let TensorFlow XLA know the CUDA path if needed using ```XLA_FLAGS=--xla_gpu_cuda_data_dir=```

<br/>

#### 4. Training the models

The models can be trained as follows:

```bash
./run_training.sh SplitterNet 20 16 [1,1,1,1] [1,1,1,1] path/to/train/image/patches/ path/to/test/images/ 5 path/to/pretrained/model 
```

where SplitterNet is the chosen denosing model, 20 is the number of epochs, 16 the batch size, the [1,1,1,1] [1,1,1,1] the number of encoder respectively decoder steps and the number of blocks for each step, followed by the path to the training patches as well as to the test images, by 5 the number of filters is given computed to the power of two, lastly the path to the pre-trained model is given.
If there is no pre-trained model, use the value None.

</br>

The final model is automatically evaluated at the end of the training process. If there is the need of evaluating models manually using the GPU it can be done by using the run_evaluation.sh script.

```bash
./run_evaluation.sh /path/to/model/ evaluate_saved_model /path/to/test/data/
```

<br/>

#### 5. Folder Creation

When training the models as shown above, there is a separate directory that is created each time with a name following this convention: 
```bash
${Model Name}_${timestamp}_e${Epochs}_bs${Batch Size}_fe${Filter Number}
```
Inside this folder you find a snapshot of the parent folder at the point when the code was executed, two log files including an .err and a .out file, a checkpoints folder where model checkpoints are saved, a trained_model folder where the final trained model is saved.

When manually evaluating as shown above a directory with the following naming convention is created:
```bash
evaluation_${timestamp}

```
Contains the respective log files as well as a snapshot of the parent folder.

#### 6. File Description

>```models/```            &nbsp; - &nbsp; different denoising models including the MoDeNet and the SplitterNet <br/>
>```data_preprocessing/``` &nbsp; - &nbsp; a set of functions as a jupyter notebook that can be used to create the patches for training <br/>

>```train.py```           &nbsp; - &nbsp; training logic <br/>
>```utils.py```           &nbsp; - &nbsp; auxiliary functions <br/>
>```evaluate.py```        &nbsp; - &nbsp; the evaluation logic <br/>
>```dataloader.py```      &nbsp; - &nbsp; the data loading logic <br/>
>```converter.py```       &nbsp; - &nbsp; code for converting the model to TensorFlow Lite. <br/>
>```environment.txt```    &nbsp; - &nbsp; environment details. <br/>
>```run_training.sh```    &nbsp; - &nbsp; commands to train on a SLURM cluster. <br/>
>```run_evaluation.sh```  &nbsp; - &nbsp; commands to evaluate on a SLURM cluster. <br/>


Inside the models folder, you find the following models:
>```SplitterNet.py```             &nbsp; - &nbsp; The new SplitterNet. <br/>
>```Dynamic_PlainNet.py```        &nbsp; - &nbsp; Dynamic implementation of the PlainNet as proposed in [NAFNet paper](https://arxiv.org/pdf/2204.04676v4.pdf) <br/>
>```Dynamic_UNet_simple.py```     &nbsp; - &nbsp; Dynamic implementation of a simple UNet <br/>
>```Megvii.py```                  &nbsp; - &nbsp; The Model proposed by Megvii research in [MAI2021](https://arxiv.org/pdf/2105.08629v1.pdf) <br/>
>```MoDeNet.py```                 &nbsp; - &nbsp; The new MoDeNet, also in a dynamical implementation <br/>
>```NOAHTCV.py```                 &nbsp; - &nbsp; The Model proposed by NOAHTCV in [MAI2021](https://arxiv.org/pdf/2105.08629v1.pdf) <br/>
>```PlainNet.py```                &nbsp; - &nbsp; Implementation of the PlainNet as proposed in [NAFNet paper](https://arxiv.org/pdf/2204.04676v4.pdf) <br/>

The term dynamic refers to the fact that you are able to give specific numbers of blocks per U-Net level as well as dictate the number of levels, you can pass the configurations of e.g. ```[2,2,4,8], [2,2,2,2]``` or ```[1,1], [1,1]``` to build custom models.

<br/>

#### 7. Model Conversion

In order to convert a model to TensorFlow Lite, add the model code or load the model in the ```converter.py``` file and call it as follows:
```bash
python converter.py
```

You may need to set ```CUDA_VISIBLE_DEVICES="" ```.

<br/>

#### 8. Local Execution

For running the code locally, which is not recommended use:
```bash
python train.py MoDeNet 20 16 None [1,1,1,1] [1,1,1,1] path/to/train/image/patches path/to/test/images/ path/to/test/images/ 5 path/to/pretrained/model 5 None
```

and for evaluation:
```bash
python evaluate.py /path/to/model/ evaluate_saved_model /path/to/test/data
```

You may need to set ```CUDA_VISIBLE_DEVICES="" ```.

