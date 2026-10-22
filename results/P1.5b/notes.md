# P1.5b notes

## Epoch-1 cutoff
floor(207000000000 / 2049) = 101024890; cutoff = floor(0.95 * 101024890) = 95973645

## Mapping test
passed=True
median drop_pred=0.1530, median drop_placebo=0.0492
bootstrap CI (paired) of median(pred)-median(placebo):
[0.0402, 0.1837]
fraction drop_pred > drop_placebo: 0.579

## Verified recollection
n=2 (infgram exclusions=65); sha256=e79ff7b25ef2534c739c33d4e980140a85b303ab450ab85469ad3630a8b518f7
