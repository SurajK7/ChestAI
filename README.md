# ChestAI

## Currently working on training diverse classifiers to create a strong ensemble

## Single Model avg AUC score for SOTA paper 0.894

## Uncertainity Labels
U-Ones: Map all uncertain labels to positive

U-Zeros: Map all uncertain labels to negative

U-Mixed: Map uncertain labels to either positive or negative according to previous results

## Transforms
Rotation, Lighting, Warping

## Abbreviations 
FE: Feature Extraction

FT: Fine Tuning

score: Avg AUC Score

## Baseline U-Mixed (the best approach)
DenseNet121, WeightDecay, Dropout, One Cycle Policy, Transforms

model0: score(FE): 0.891; score(FT): **0.89**

model1(Mixed Precision): score(FE): **0.891**

model2: score(FE): 0.889

model3: score(FE): 0.882

ensemble of above 4 models: **0.895** (The increment is marginal owing to the similar architectures of models)

## Baseline U-Ones 
DenseNet121, WeightDecay, Dropout, One Cycle Policy, Transforms

score(FE): 0.87

## Baseline U-Zeros 
DenseNet121, WeightDecay, Dropout, One Cycle Policy, Transforms

score(FE): 0.875

## Progressive (doesn't help in our case)
DenseNet121, U-Mixed, WeightDecay, Dropout, One Cycle Policy, Transforms, scaled from 160 to 320 px

score(160px, FE):  0.878; score(160px, FT): 0.879; score(320px, FE): 0.887

## Label Smoothing (doesn't help in our case)
DenseNet121, U-Mixed, WeightDecay, Dropout, One Cycle Policy, Transforms, Mapped uncertain 0s to 0.2s and 1s to 0.8s

score(FE): 0.878

## Regularization (doesn't help in our case)
DenseNet121, U-Mixed, One Cycle Policy, Transforms

score: NA

