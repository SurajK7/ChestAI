# ChestAI

## Uncertainity Labels
U-Ones: Map all uncertain labels to positive

U-Zeros: Map all uncertain labels to negative

U-Mixed: Map uncertain labels to either positive or negative according to previous results

## Transforms
Rotation, Lighting, Warping

## Baseline U-Ones 
DenseNet121, WeightDecay, Dropout, One Cycle Policy, Transforms

## Baseline U-Zeros 
DenseNet121, WeightDecay, Dropout, One Cycle Policy, Transforms

## Baseline U-Mixed 
DenseNet121, WeightDecay, Dropout, One Cycle Policy, Transforms

## Progressive
DenseNet121, U-Mixed, WeightDecay, Dropout, One Cycle Policy, Transforms, scaled from 160 to 320 px

## Label Smoothing
DenseNet121, U-Mixed, WeightDecay, Dropout, One Cycle Policy, Transforms, Mapped uncertain 0s to 0.2s and 1s to 0.8s

## Regularization
DenseNet121, U-Mixed, One Cycle Policy, Transforms
