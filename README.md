# interpret-lesion-unc
Multiple sclerosis cortical lesion segmentation model on MP2RAGE - DE and MCDP uncertainty quantification - Lesion-scale uncertainty - Instance-wise uncertainty - Interpretability of lesion-scale uncertainty - Interpretability of instance-wise uncertainty

This repo contains implementations for this manuscript: [Arxiv](https://arxiv.org/abs/2407.05761). 

## Requirements

The conda environment has been exported into requirements.txt. Python version 3.8.10.

## Cortical lesion segmentation model

Two models are adopted based on a similar 3D U-Net architecture: deep ensemble (DE) and Monte Carlo Dropout (MCDP).

Architecture details can be found in the [MONAI library documentation (v0.9.0)](https://docs.monai.io/en/0.9.0/networks.html#unet). 

Hyperparameters:

| Hyperparameter      | Description                                |
|---------------------|--------------------------------------------|
| **Training** |
| loss              | Focal weighted loss (weight 2e5, gamma 2)     |
| scheduler         | Plateau (gamma 0.5, step_size 10)               |
| initial learning rate     | 1e-2                   |
| warmup iters      | 400             |
| minimum learning rate      | 1e-7                 |
| early stopping         | tolerance 1e-10, patience 100         |
| epoch saving       | on maximal validation nDSC score          |
| validation every n epochs | 1 |
| n_epochs          | 400                  |
| batch size        | 8                    |
| **Data processing** |
| input subvolumes shape | (96, 96, 96) |
| subvolumes from a single scan         | 32     |
| augmentation strategy        | [get_cltrain_transforms](utils/transforms.py)           |
| inference preprocessing | intensity normalization |
| inferer | [sliding window](https://docs.monai.io/en/0.9.0/inferers.html?highlight=SlidingWindowInferer#slidingwindowinferer), 25% overlap, gaussian weighting |
| **Model** |
| architecture | 3D U-Net
| spatial_dims      | 3     |
| in_channels       | 96        |
| out_channels      | 1        |
| channels          | (32, 64, 128, 256)             |
| strides           | 2                          |
| norm              | 'batch'      |
| num_res_units     | 0         |
| dropout           | 0.1 for MCDP, None for DE          |



Model training code: [train.sh](bash_scripts/train.sh).

Model inference code: [predict.sh](bash_scripts/predict.sh).


**Model performance:**

Model performance:

| Set                             | Model    | DSC                               | F1-score (iou\_adj)               | Precision (iou\_adj)              | Recall (iou\_adj)                 | ECE bin                           |
|---------------------------------|----------|-----------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|
| Train          | DE       | $0.640_{[0.599, 0.673]}$          | $0.798_{[0.752, 0.832]}$          | $0.878_{[0.829, 0.913]}$          | $0.760_{[0.711, 0.800]}$          | $0.004_{[0.004, 0.004]}$          |
|                                 | MCDP     | $0.518_{[0.470, 0.559]}$          | $0.669_{[0.608, 0.720]}$          | $0.742_{[0.676, 0.796]}$          | $0.642_{[0.581, 0.698]}$          | $0.005_{[0.005, 0.005]}$          |
| Val            | DE       | $0.541_{[0.428, 0.623]}$          | $0.676_{[0.598, 0.790]}$          | $0.751_{[0.621, 0.847]}$          | $0.642_{[0.559, 0.770]}$          | $0.004_{[0.004, 0.005]}$          |
|                                 | MCDP     | $0.457_{[0.331, 0.538]}$          | $0.629_{[0.506, 0.752]}$          | $0.732_{[0.544, 0.851]}$          | $0.574_{[0.479, 0.724]}$          | $0.005_{[0.005, 0.006]}$          |
| Test        | DE       | $0.385_{[0.308, 0.452]}$          | $0.549_{[0.442, 0.638]}$          | $0.622_{[0.496, 0.730]}$          | $0.555_{[0.446, 0.648]}$          | $0.004_{[0.004, 0.004]}$          |
|                                 | MCDP     | $0.338_{[0.266, 0.404]}$          | $0.485_{[0.382, 0.575]}$          | $0.588_{[0.460, 0.702]}$          | $0.488_{[0.381, 0.587]}$          | $0.005_{[0.005, 0.005]}$          |


# Lesion uncertainty regression

Lesion uncerainty computation: [precompute_les_uncs.sh](bash_scripts/precomute_les_uncs.sh) -- "ddu" measure in the saved dictionaty.

Uncertainty regression model: [Elasticnet](uncertainty_regression/elasticnet_repeated_cv.py).

Uncertainty regression quality:

|       Uncertainty         | CV Only IoU$_{adj}$       | CV No IoU$_{adj}$           | CV All                 | Test Only IoU$_{adj}$    | Test No IoU$_{adj}$      | Test All                 |
|---------------------|------------------------|--------------------------|---------------------|------------------------|--------------------------|--------------------------|
| DE                  | $0.520_{\pm 0.006}$    | $0.598_{\pm 0.004}$      | $0.661_{\pm 0.004}$ | $0.431_{\pm 0.001}$    | $0.512_{\pm 0.002}$      | $0.632_{\pm 0.004}$ |
| MCDP                | $0.393_{\pm 0.006}$    | $0.589_{\pm 0.014}$      | $0.604_{\pm 0.013}$ | $0.261_{\pm 0.003}$ | $0.425_{\pm 0.013}$ | $0.494_{\pm 0.004}$ |

# References

```
@misc{molchanova2024,
      title={Interpretability of Uncertainty: Exploring Cortical Lesion Segmentation in Multiple Sclerosis}, 
      author={Nataliia Molchanova and Alessandro Cagol and Pedro M. Gordaliza and Mario Ocampo-Pineda and Po-Jui Lu and Matthias Weigel and Xinjie Chen and Adrien Depeursinge and Cristina Granziera and Henning Müller and Meritxell Bach Cuadra},
      year={2024},
      eprint={2407.05761},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2407.05761}, 
}
@inproceedings{molchanovaiee2023,
   title={Novel Structural-Scale Uncertainty Measures and Error Retention Curves: Application to Multiple Sclerosis},
   url={http://dx.doi.org/10.1109/ISBI53787.2023.10230563},
   DOI={10.1109/isbi53787.2023.10230563},
   booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
   publisher={IEEE},
   author={Molchanova, Nataliia and Raina, Vatsal and Malinin, Andrey and La Rosa, Francesco and Muller, Henning and Gales, Mark and Granziera, Cristina and Graziani, Mara and Cuadra, Meritxell Bach},
   year={2023},
   month=apr 
}
@misc{molchanova2024preprint,
      title={Structural-Based Uncertainty in Deep Learning Across Anatomical Scales: Analysis in White Matter Lesion Segmentation}, 
      author={Nataliia Molchanova and Vatsal Raina and Andrey Malinin and Francesco La Rosa and Adrien Depeursinge and Mark Gales and Cristina Granziera and Henning Muller and Mara Graziani and Meritxell Bach Cuadra},
      year={2024},
      eprint={2311.08931},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.08931}, 
}
```

# License

This work is protected by [CC BY-NC-SA 4.0 License](LICENSE). This license enables reusers to distribute, remix, adapt, and build upon the material in any medium or format for noncommercial purposes only, and only so long as attribution is given to the creator. If you remix, adapt, or build upon the material, you must license the modified material under identical terms.
