# PyTorch Implementation of PPnP Paper

**Reference:**  
[Tao Hong](https://hongtao-argmin.github.io), Xiaojian Xu, Jason Hu, and [Jeffrey A. Fessler](https://web.eecs.umich.edu/~fessler/), [Provable Preconditioned Plug-and-Play Approach for Compressed Sensing MRI Reconstruction](https://arxiv.org/abs/2405.03854), IEEE Transactions on Computational Imaging, vol. 10, pp. 1476–1488, Oct. 2024.  

## Getting Started
1. Install the required packages using the provided `PPnP.yml` file.  
2. **`demoReco.py`** – contains a demo example for setting up the experiment and running the algorithm.  
3. **`optalg_Tao_Pytorch.py`** – implements the optimization algorithms and several utility functions.  
4. **`optpoly.py`** – provides functions to compute coefficients for the Chebyshev polynomial preconditioner. (Requires an additional package: [Chebyshev](https://github.com/mlazaric/Chebyshev)).  

## Models
- Download trained models from [Google Drive](https://drive.google.com/drive/folders/1QN7t0l3PnzHqvx9d-npONiF8Sz946Oi_?usp=share_link).  
- Alternatively, you can train your own models.  

## Key Insights from the Paper
1. Preconditioners can be effectively incorporated into the PnP framework to accelerate reconstruction.  
2. Preconditioning not only improves convergence speed but also enhances image quality.  

## Contact
If you find bugs or encounter difficulties using this implementation, please contact me at **tao.hong@austin.utexas.edu**.  

## Note
If you are interested in discussing our work further, feel free to reach out as well. 