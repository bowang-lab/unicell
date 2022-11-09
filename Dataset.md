# Dataset



CellUniverse consists of our new collection of labeled blood cell images and six existing public cell datasets, including fluorescent images with various immunostaining, brightfield images with Jenner-Giemsa staining as well as label-free images (e.g., phase-contrast, differential interference contrast (DIC) images).



| Name      | Microscope                 | Staining       | Origin           | Tonality    | \# Annotations |
| --------- | -------------------------- | -------------- | ---------------- | ----------- | -------------- |
| Cellpose  | Various                    | Various        | Various tissues  | Two-channel | 71,881         |
| TissueNet | Fluorescent                | Immunostaining | Various tissues  | Two-channel | 1,179,772      |
| Omnipose  | Phase-contrast/Fluorescent | N/A            | Various bacteria | Grey        | 41753          |
| SegPC     | Brightfield                | Jenner-Giemsa  | Multiple Myeloma | RGB         | 2459           |
| YeaZ      | Phase-contrast/Brightfield | N/A            | Yeast            | Grey        | 2993/23046     |
| YeastMate | DIC/Brightfield            | N/A            | Yeast            | Grey        | 10206/7885     |
| Ours      | Brightfield                | Jenner-Giemsa  | Bone Marrow      | RGB         | 131,327        |



All the public dataset can be downloaded from the original paper. Our annotated blood cell dataset can be downloaded at https://drive.google.com/file/d/1QF3nb3exEHMN7h-blXoRKgWxEf0tAieX/view?usp=share_link. The images were from this [kaggle dataset](https://www.kaggle.com/datasets/sebastianriechert/bone-marrow-slides-for-leukemia-prediction).