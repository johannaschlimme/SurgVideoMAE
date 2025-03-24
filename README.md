# VideoMAEv2-based Pretraining and Fine-Tuning for Ophthalmic Phase Recognition

## Overview
This repository is based on [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2). It focuses on pretraining a transformer-based video model on **OphNet data** and fine-tuning it for **phase recognition in cataract surgery videos**.

### Model

VideoMAEv2 uses a **dual masking strategy** to learn video representations in a self-supervised manner.

![VideoMAEv2 Model](docs/VideoMAEv2_flowchart.png)

This approach is used in our pretraining step with **OphNet data**.

## Installation

To set up the environment, use the provided Conda environment file:

```sh
conda env create -f environment.yml
conda activate videomae
```

If needed, for OpenCV compatibility, install system libraries:

```sh
apt-get update && apt-get install -y libgl1 libglib2.0-0
```

## Data Preprocessing

### Preprocessing for OphNet (Pretraining)
Run the script to preprocess **OphNet data** for self-supervised pretraining:

```sh
python data/preproc_pretrain_ophnet.py --ophnet_video_dir /path/to/ophnet_video_files --ophnet_annotations_csv /path/to/ophnet_annotations --output_dir path/to/preprocessed_output_directory
```

### Preprocessing for Cataract Data (Fine-Tuning)
Currently, fine-tuning for cataract phase recognition is under development.

## Running Pretraining
To run VideoMAE pretraining on **OphNet data**, execute:

```sh
bash scripts/ophnet_pretrain.sh
```

## Running Fine-Tuning for Phase Recognition
Currently, fine-tuning for cataract phase recognition is under development.

## Ongoing Work
- Fine-tuning the model for **phase recognition** in cataract surgery videos.
- Hyperparameter tuning for improved performance.

## Contact

This project was developed as part of research at Ludwig-Maximilians-University Munich.

- Author: Johanna Schlimme
- Supervisors: Mina Rezaei, Dr. Mohammad Eslami
- Affiliation: Ludwig-Maximilians-University Munich
- Email: johanna.schlimme@campus.lmu.de  

## Acknowledgement

This repository is built upon [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2). If you use this work, please consider citing the original VideoMAEv2 paper:

```
@InProceedings{wang2023videomaev2,
    author    = {Wang, Limin and Huang, Bingkun and Zhao, Zhiyu and Tong, Zhan and He, Yinan and Wang, Yi and Wang, Yali and Qiao, Yu},
    title     = {VideoMAE V2: Scaling Video Masked Autoencoders With Dual Masking},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {14549-14560}
}

@misc{videomaev2,
      title={VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking},
      author={Limin Wang and Bingkun Huang and Zhiyu Zhao and Zhan Tong and Yinan He and Yi Wang and Yali Wang and Yu Qiao},
      year={2023},
      eprint={2303.16727},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

