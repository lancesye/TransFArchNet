您说得对，我之前遗漏了部分注释内容。以下是完整的修正版 README.md：

```markdown
# TransFArchNet

## Citation

If you find this work useful for your research, please cite our paper:

> Guoye Lin, Shuo Yang, Yangfan Chen, Qing Xu, Pew-Thian Yap, Zhaoqiang Yun, Qianjin Feng.  
> **TransFArchNet: Predicting dental arch curves based on facial point clouds in personalized panoramic X-ray imaging**.  
> *Expert Systems with Applications*, Volume 270, 2025, 126577.  
> DOI: [10.1016/j.eswa.2025.126577](https://doi.org/10.1016/j.eswa.2025.126577)

📄 [View on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S095741742500199X)

## Abstract

> Traditional panoramic X-ray imaging, constrained by a fixed scanning path aligned with a preset dental arch curve, often results in image distortions such as blurring, magnification, or shrinking, particularly when structures lie outside the focal layer, leading to diagnostic inaccuracies. To address this, we propose **TransFArchNet**, a novel personalized panoramic scanning solution that predicts individualized dental arch curves from scanned facial point clouds. Utilizing multi-scale transformer layers and feature selection modules, the network is able to learn both global and local facial geometry. Our method is robust, incorporating a noise-filtering step, and an auxiliary confidence network that emphasizes stable facial points to improve curve-fitting accuracy, particularly for jawbone-located dental arches, distant from the soft-tissue facial point clouds. Experimental validation using cone-beam computed tomography (CBCT)-derived facial point clouds shows that TransFArchNet achieves a mean 2D prediction error of 1.50 mm, outperforming PointNet++'s error of 1.67 mm and smaller than the dimensions of a small tooth. Besides, the promising results on real facial predictions suggest potential clinical applications.

## Repository

🔗 **Code**: [https://github.com/lancesye/TransFArchNet](https://github.com/lancesye/TransFArchNet)

---

## Installation

Install the required Python libraries:

```

---

## Code Execution Order

### 1. Data Preprocessing and Saving

> For `all_point in -all.txt`, dental arch points need to be preprocessed and uniformly sampled to 512 points.  
> **Example:**  
> ```python
> ctr_point = get_line_data("1000813648_20180116-ctr.txt")
> all_point = sample_ctr_to_densePoint(ctr_point, sample_point=512)
> # Save all_point as "1000813648_20180116-all.txt"
> ```

```bash
python 1_data_preprocess.py
```

### 2. Train, Validate, and Test the Model

```bash
python 2_train_TransArchNet.py
```

### 3. Inference on External Data

**Data Path Parameters**

- **Test data**  
  `mesh_folder = r'CBCT_Mesh_data\test_data'`

- **Manually specify the best model weight path**  
  `model_path = TransFArchNet_checkpoints/checkpoints_best.pth`

```bash
python 3_predict_TransArchNet.py
```

---

## Contact

Feel free to contact me if you have any further questions.
```
