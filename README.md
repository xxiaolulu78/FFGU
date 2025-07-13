# FORGET ME: Federated Unlearning for Face Generation Models
<img width="1574" height="480" alt="image" src="https://github.com/user-attachments/assets/798eab81-a733-4d35-a981-7ef696f9e199" />

## Setup

    conda env create -f environment.yaml/environment2.yaml



## Dataset
This project utilizes several large-scale public face datasets. You are required to download them manually from their official sources.
The core datasets are:[**VGG-Face2**](https://github.com/ox-vgg/vgg_face2) 、 [**CelebA-HQ**](https://paperswithcode.com/dataset/celeba-hq) 、 [**CASIA-WebFace**](https://paperswithcode.com/dataset/casia-webface)
After downloading the datasets, you need to perform a custom partitioning to simulate a federated learning environment. The data should be organized into **128 client datasets**, with the constraint that each client possesses images of only a **single, unique face identity**.
Before the forgetting operations, all face images must be pre-processed and augmented. The data augmentation should follow the methodology used for processing face images in the **[Deep3DFaceReconstruction](https://github.com/sicxu/Deep3DFaceRecon_pytorch)** project. This ensures that the input data is normalized and prepared in a consistent manner for the model.



## Unlearn Train

    pyhton Fed_Unlearn_gan.py
    pyhton Fed_Unlearn_dm.py

## Acknowledgements
We would like to express our gratitude to the authors of the following projects for their invaluable contributions and for open-sourcing their work:[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)、[GOAE](https://github.com/jiangyzy/GOAE) 、[EG3D](https://github.com/NVlabs/eg3d)、


