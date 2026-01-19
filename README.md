# Solar Flare Forecasting

Modeling full-disk solar magnetic field features and forecasting solar flares using deep convolutional neural networks.

## Work

Four deep convolutional neural network architectures—AlexNet, VGGNet, Inception, and ResNet—were constructed based on full-disk solar magnetograms to enable solar flare forecasting.

## Results

The models were evaluated using flare activity data from 2020 to 2024. The evaluation metrics are summarized below:

| Threshold | Model        | TSS               | HSS               | Accuracy          | Precision         | Recall              | FAR               |
| ---- | ----------- | ----------------- | ----------------- | ----------------- | ----------------- | ------------------- | ----------------- |
| ≥C   | AlexNet     | $0.747 \pm 0.056$ | $0.756 \pm 0.025$ | $0.891 \pm 0.006$ | $0.908 \pm 0.043$ | $0.931  \pm 0.0043$ | $0.092 \pm 0.043$ |
|      | VGG-16      | $0.796 \pm .009$  | $0.797 \pm 0.013$ | $0.908 \pm 0.007$ | $0.928 \pm 0.004$ | $0.932 \pm 0.013$   | $0.072 \pm0.004$  |
|      | InceptionV3 | $0.796 \pm 0.012$ | $0.794 \pm 0.010$ | $0.906 \pm 0.005$ | $0.930 \pm 0.008$ | $0.926 \pm 0.010$   | $0.070 \pm 0.008$ |
|      | ResNet      | $0.787 \pm 0.011$ | $0.766 \pm 0.038$ | $0.890 \pm 0.023$ | $0.948 \pm 0.033$ | $0.883 \pm 0.070$   | $0.052\pm0.033$   |
| ≥M   | AlexNet     | $0.529 \pm 0.043$ | $0.465 \pm 0.029$ | $0.744 \pm 0.041$ | $0.561 \pm 0.071$ | $0.813 \pm 0.160$   | $0.439 \pm 0.071$ |
|      | VGG-16      | $0.575 \pm 0.021$ | $0.528 \pm 0.020$ | $0.784 \pm 0.017$ | $0.604 \pm 0.031$ | $0.796 \pm 0.057$   | $0.396 \pm 0.031$ |
|      | InceptionV3 | $0.563\pm0.032$   | $0.497\pm0.027$   | $0.762\pm0.023$   | $0.570\pm0.037$   | $0.829\pm0.082$     | $0.430\pm0.037$   |
|      | ResNet      | $0.602\pm0.012$   | $0.517\pm0.031$   | $0.765\pm0.026$   | $0.568\pm0.036$   | $0.889\pm0.054$     | $0.432\pm0.036$   |
| ≥X   | AlexNet     | $0.405\pm0.015$   | $0.083\pm0.026$   | $0.669\pm0.131$   | $0.078\pm0.016$   | $0.738\pm0.157$     | $0.922\pm0.016$   |
|      | VGG-16      | $0.350\pm0.031$   | $0.100\pm0.027$   | $0.733\pm0.075$   | $0.090\pm0.019$   | $0.570\pm0.110$     | $0.910\pm0.019$   |
|      | InceptionV3 | $0.445\pm0.034$   | $0.085\pm0.025$   | $0.655\pm0.103$   | $0.079\pm0.015$   | $0.795\pm0.141$     | $0.921\pm0.015$   |
|      | ResNet      | $0.455\pm0.033$   | $0.089\pm0.031$   | $0.653\pm0.130$   | $0.081\pm0.019$   | $0.807\pm0.162$     | $0.919\pm0.019$   |

Finally, the trained models were packaged into a graphical user interface application using PySide6, as shown in the figure below:

![4-3](https://github.com/user-attachments/assets/906bed8f-dca3-4c78-90ee-4abf02ee71b7)
