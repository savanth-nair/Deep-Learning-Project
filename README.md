
# Transfer Learning with Convolutional Neural Networks on Food101 Dataset

📌 **Project Overview**  
This project explores **transfer learning techniques** by applying pre-trained **CNN architectures** (MobileNet V3, NasNet, and GoogLeNet) on the **Food101** dataset to classify 101 food categories.  
The objective is to leverage transfer learning to enhance model performance while reducing training time.

---

📖 **Introduction**  
The Food101 dataset is a challenging benchmark consisting of 101,000 images across 101 food categories.  
This project demonstrates how **transfer learning** using **MobileNet V3**, **NasNet**, and **GoogLeNet** can efficiently adapt pre-trained models to new tasks through head replacement, parameter freezing, fine-tuning, and optimizer adjustments.

---

🗄️ **Dataset Presentation**
- **Dataset:** [Food101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- **Images:** 101,000 total (750 training and 250 testing images per class)
- **Image Size:** Rescaled to a maximum side length of 512 pixels
- **Preprocessing:**  
  - Normalization  
  - Data Augmentation: Random cropping and horizontal flipping

---

🏗️ **Architectures Used for Transfer Learning**

| Model         | Key Features |
|:--------------|:-------------|
| **MobileNet V3** | Depthwise separable convolutions, squeeze-and-excitation blocks, mobile-optimized |
| **NasNet**     | Neural Architecture Search optimized, uses depthwise-pointwise convolutions |
| **GoogLeNet**  | Inception modules (1x1, 3x3, 5x5 convolutions), depth concatenation |

**Common Adaptations:**
- Classifier Head replaced with custom layers (dense layers + softmax)
- Input resizing and augmentation
- CrossEntropyLoss used
- Adam or SGD Optimizers

---

🔧 **Transfer Learning Training Process**
- **Parameter Freezing:** Early layers frozen to retain ImageNet-learned features.
- **Custom Head Design:** Fully connected layers adapted for 101-class output.
- **Optimizers:**
  - MobileNetV3: SGD with momentum
  - NasNet and GoogLeNet: Adam optimizer
- **Loss Function:** CrossEntropyLoss
- **Batch Training:** Mini-batch gradient descent
- **Epoch-wise Validation:** Monitoring accuracy and loss each epoch

---

📈 **Performance Analysis of Pre-trained Models**

| Metric | MobileNet V3 | NasNet | GoogLeNet |
|:------|:-------------|:------|:---------|
| Initial Training Accuracy | 44.65% | 1.16% | 1.16% |
| Final Training Accuracy | 74.13% | 1.48% | 1.48% |
| Initial Validation Accuracy | 46.07% | 1.08% | 1.08% |
| Peak Validation Accuracy | 47.06% | 1.20% | 1.20% |
| Validation Loss (Final) | 2.3311 | 4.6151 | 4.6151 |

✅ **MobileNet V3 outperformed** NasNet and GoogLeNet significantly.  
⚠️ NasNet and GoogLeNet suffered from convergence issues.

---

🛠️ **Fine-Tuning Approach on MobileNet V3**

| Aspect | Details |
|:------|:--------|
| Optimizer | SGD with Momentum (lr=0.001, momentum=0.9) |
| Loss Function | CrossEntropyLoss |
| Learning Rate Scheduler | ReduceLROnPlateau (patience=3, factor=0.1) |
| Early Stopping | Patience=5 |
| Data Augmentation | Random cropping, horizontal flipping |
| Checkpointing | Saving best model |

---

📊 **Fine-Tuned Model Results (MobileNet V3)**

| Metric | Value |
|:------|:------|
| Final Training Accuracy | 8.88% |
| Peak Validation Accuracy | 6.86% |
| Final Validation Loss | 4.5749 |
| Test Accuracy | 7.26% |
| Test Loss | 4.5718 |

**Observations:**
- Training and validation accuracy gradually improved over epochs.
- Moderate overfitting observed.
- Fine-tuned model achieved a test accuracy of 7.26%.

---

💬 **Comparison: Transfer Learning (Part A) vs Fine-Tuning (Part B)**
- **Part A (transfer learning only)** had much better performance (~47% validation accuracy).
- **Part B (fine-tuning)** had limited improvement (~7% test accuracy).
- Suggests that **early transfer learning without modifying many layers worked better**.

---

⚠️ **Remaining Issues**
- Low final accuracy after fine-tuning
- High initial loss indicating dataset complexity
- Generalization to unseen data still a challenge

---

💡 **Recommendations for Improvement**
- Try **different optimizers** (e.g., AdamW, Ranger)
- **Lower initial learning rate** further
- **Unfreeze more layers** during fine-tuning
- Add **batch normalization** or **advanced regularization**
- Explore **longer training** or **ensemble models**

---

📬 **References**
1. [Colab Notebook - Part A](https://colab.research.google.com/drive/1jcvLGAWhQUjw8zrSXdZktPlROM1uaKYT?usp=drive_link)  
2. [Colab Notebook - Part B](https://colab.research.google.com/drive/1okCUuDSIeyKylmoJRQ85POdaqGcuuFY7?usp=drive_link)

---

🚀 **Conclusion**  
Transfer learning using MobileNet V3 showed strong performance on Food101, while NasNet and GoogLeNet underperformed. Fine-tuning requires careful adjustment of learning rates, optimizers, and freezing strategies. Future enhancements could further boost accuracy through better model adaptation and advanced techniques like Neural Architecture Search or ensembling.

---

📜 **License**  
This project is part of an academic assignment and subject to university guidelines.

---

🚀 **Happy Learning!**
