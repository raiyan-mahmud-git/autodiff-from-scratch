# Experiment notes

These observations describe only the configured architecture, learning rates, epoch budget, split, and five seeds used by the scripts. They are not claims about every learning rate, dataset, or model size.

## Optimizers

At the configured rates and five-epoch limit, SGD did not reach the 0.95 validation-accuracy target, while momentum SGD and Adam did; Adam reached it sooner on average in these runs. Adam also had the highest final validation accuracy here, but different fixed learning rates and a short run mean this does not establish that Adam is generally the best optimizer.

## Dropout

Dropout reduced the mean train-minus-validation accuracy gap, while the final validation accuracies were within their across-seed variation, so this run shows no clear validation-accuracy difference. This result applies to the tested dropout probability and training budget; other regularization settings or longer training could behave differently.

## Initialization

He initialization produced much lower final training loss and higher validation accuracy than naive initialization in these runs, and all runs remained numerically stable. This supports the value of a scale suited to ReLU layers in this setup, but it does not prove the same result for other optimizers, architectures, or data.

## Handwriting robustness check

Small affine changes to the training images improved mean validation accuracy from 0.9673±0.0023 to 0.9709±0.0015 across the five seeds, and mean validation recall for digit 9 from 0.9413±0.0067 to 0.9552±0.0025. The seed-zero affine checkpoint is used by the web demo; this comparison uses validation data only and does not establish accuracy on human drawings or on the test set.

## Digit 9 calibration

On the seed-zero affine checkpoint, a validation-only class bias increased 9 recall from 0.9543 to 0.9739 while validation accuracy changed from 0.9734 to 0.9732; 9 precision changed from 0.9585 to 0.9392. The script chooses the highest 9 recall among offsets whose overall validation accuracy is within 0.001 of the best grid result; this trades some false 9 predictions for better sensitivity and is not a substitute for measuring human-drawing accuracy.
