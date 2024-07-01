# Advancing Ulcerative Colitis Diagnosis: A Transformer-Based Approach to Mayo Endoscopic Classification

### Halil Ibrahim Donmezbilek

## Introduction
Ulcerative Colitis (UC) is one of the inflammatory bowel diseases that leads to inflammation and sores forming in the digestive system. It requires regular check-ups. Doctors use  Mayo Endoscopic Score (MES) to how severe the UC ranges from 0 (normal) to 3 (severe), with increasing severity indicated by symptoms such as erythema, friability, erosions, and spontaneous bleeding.

This project aims to use a transformer model to make the scoring of UC inflammation more accurate and consistent. By comparing the transform model with two baseline non-transformer models, the project aims to show that the transformer model can better identify the severity of inflammation from these images, which could help doctors treat their patients more effectively.

## Literature
In the study "Novel Deep Learning–Based Computer-Aided Diagnosis System for Predicting Inflammatory Activity in Ulcerative Colitis," [^1], researchers developed an AI model to enhance the accuracy of scoring inflammatory activity in patients with ulcerative colitis using endoscopic images and videos. The dataset consisted of 5,875 endoscopic images and 20 complete videos collected from 332 patients, applying the ResNet50 model pre-trained on ImageNet. The model demonstrated F-scores of 87.14%, 72.05%, 86.10%, and 92.06% for Mayo scores 0, 1, 2, and 3, respectively.

In the study "IEViT: An Enhanced Vision Transformer Architecture for Chest X-ray Image Classification" [^2], the authors introduced a new model, the Input Enhanced Vision Transformer (IEViT), designed to classify chest X-ray images more accurately. This model outperformed the original Vision Transformer (ViT) across various metrics. Notably, IEViT achieved F1-scores up to 100%, showcasing enhanced performance and generalization capabilities over existing models. This proposed model promises significant assistance in diagnosing diseases from chest X-rays.

## Exploratory Data Analysis (EDA)

![NumberOfImagesbyLabelAndDataset](https://github.com/halildonmezbilek/MayoEndoscopicClassification-SwinTransformer/assets/40296559/803b4024-7a72-4e78-8fb0-81681307b15e)

**Figure 1:** Image Class Distribution

The exploratory data analysis shows (as shown in **Figure 1**) how images are distributed across four severity levels of Ulcerative Colitis, labeled as Mayo 0 to Mayo 3, and dataset splits. This distribution shows a common skew often seen in health data, where less severe cases are more frequently represented. In the training dataset, Mayo 0, which represents normal conditions, has the most images at 4,144, showing a strong focus on cases without inflammation. Mayo 1 follows with 2,070 images, 861 and 596 for Mayo 2 and Mayo 3, respectively.

## Naive Model
The first baseline model was a naive model, which predicted the most frequent class observed in the training dataset. This simple model serves as a foundational comparison point to assess the effectiveness of more sophisticated algorithms.

The naive model predicts by first determining the most common class in the training data and then uniformly assigning this class to all instances in the validation and test datasets. The performance of this model on the validation set yielded an accuracy of 53.99% Similarly, on the test dataset, the model achieved an accuracy of 54.86% These results highlight the model's basic capability to classify based on the dominant class but also underline its insufficient in providing meaningful insights or reliable predictions across varied or balanced datasets.

## Non-Transformer Deep Neural Network Model (NTDNN)

![NonTransformerConfusionMatrix](https://github.com/halildonmezbilek/MayoEndoscopicClassification-SwinTransformer/assets/40296559/7a71421a-56d6-4fb0-9870-1b8763f9b235)

**Figure 2:** Non-Transformer Deep Network Model Test Confusion Matrix

For the second baseline model in the project, pretrained ResNet-101 architecture was utilized, a widely recognized deep learning model known for its effectiveness in image classification tasks. This model was chosen due to its proven capability in handling complex image data and its suitability for the image classification needs. The results from this non-transformer model provided a benchmark against a transformer model. The model achieved a validation Quadratic Weighted Kappa (QWK) of 0.7785 and an accuracy of 0.7237 by the 25th epoch. On the test set,(confusion matrix as shown in **Figure 2** the model achieved a QWK of 0.7602 and an accuracy of 0.6892.

## Proposed Method - SwinColorFusionNet

Swin Transformer [^3] is a transformer-based backbone model for vision and it creates layered feature maps by merging image patches as they go deeper into the hierarchy. Combining this initial global contextual information with other global information from color histograms can enhance the model's performance, leading to more accurate image classification results. The reason for choosing the color histogram is that while the MES score goes from 0 to 3, the human tissue has more color red. It may be blood or simply an indication of erosions. The color histogram stores some global information for the classification of the MES score. 

The dataset contains images with 3 color channels, and color histograms are created using 16 bins for each channel. These histogram features are concatenated with the features extracted by the Swin Transformer model. The combined features are then passed through a fully connected layer with 128 units, followed by a ReLU activation function. Finally, the output layer classifies the input into 4 classes. 

## Results of Proposed Method

| Model                    | Accuracy | QWK    |
|--------------------------|----------|--------|
| Naive Model              | 0.5486   | -      |
| Non-Transformer Model    | 0.6892   | 0.7602 |
| SwinColorFusionNet Model | 0.774    | 0.8552 |

**Table 1:** Performance Comparison

![BestSwinFusionColorNetModelConfusionMatrix](https://github.com/halildonmezbilek/MayoEndoscopicClassification-SwinTransformer/assets/40296559/e8bdf952-5e63-452f-81f9-6cf1c5fc2521)

**Figure 3:** Proposed Model Test Confusion Matrix

Weight and Biases (WANDB) is used to tune the hyperparameters of the model, these are the number of epochs, learning rate, and batch size. The model is ran 20 times to find the best settings. The best model achieved a QWK score of 0.8552 and an accuracy of 0.774. All the results are recorded on the Weight and Biases website and can be accessed online [^4].

The performance of the base models and the proposed model is shown in **Table 1**. The proposed method outperforms all other models in every performance metric. Additionally, Wilcoxon signed-rank tests confirm that the proposed model's improvements are statistically significant. The Wilcoxon test between SwinColorFusionNet and the Naive model resulted in a statistic of 0.0 and a p-value of 5.14e-113. The Wilcoxon test between SwinColorFusionNet and the NTDNN model resulted in a statistic of 31535.0 and a p-value of 3.07e-27.

The confusion matrices show that the SwinColorFusionNet model performs better than the NTDNN in classifying Mayo 0, Mayo 2, and Mayo 3, with more correct predictions and fewer mistakes. The NTDNN model only outperforms SwinColorFusionNet in classifying Mayo 1. Overall, SwinColorFusionNet is a more effective model for this dataset.

## Discussions and Conclusion

The SwinColorFusionNet model effectively enhances image classification by combining global contextual features from the Swin Transformer with global color histogram features. It consistently outperforms the Naive and NTDNN models in accuracy and QWK metrics, with statistically significant improvements.

Interestingly, the NTDNN model outperforms the SwinColorFusionNet model in classifying Mayo 1. This could be due to the NTDNN model's specific architecture being more adept at recognizing features associated with Mayo 1, possibly capturing more prominent or distinct patterns in this class. Further investigation into the feature representations learned by both models could provide more insights.

Future work could further refine this approach and explore additional global features for even better performance. 

## References
[^1]: Y. Fan, R. Mu, H. Xu, C. Xie, Y. Zhang, L. Liu, L. Wang, H. Shi,Y. Hu, J. Ren, J. Qin, L. Wang, and S. Cai, “Novel deep learning–based computer-aided diagnosis system for predicting inflammatory activity in ulcerative colitis,” Gastrointestinal Endoscopy, vol. 97, no. 2, pp. 335–346, Feb. 2023.
[^2]: G. I. Okolo, S. Katsigiannis, and N. Ramzan, “Ievit: An enhanced vision transformer architecture for chest x-ray image classification,” Computer Methods and Programs in Biomedicine, vol. 226, p. 107141, Nov. 2022.
[^3]: Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” Mar. 2021.
[^4]: H. Donmezbilek, “Wandb sweep results of swincolorfusionnet,https://wandb.ai/halil-donmezbilek/Mayo-Endoscopic-Classification-Swin-Transformer?nw=nwuserhalildonmezbilek, 2024, accessed: July 1, 2024.

