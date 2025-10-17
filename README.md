# AircraftClassifier_CNN_SVM

## Faza 1 (EDA & Pretproces)
- Učitavanje FGVC Aircraft, TOP-10 klasa, stratifikovan split (70/15/15)
- Generator(i): resize 128×128, normalizacija, augmentacija (train)
- Splitovi snimljeni u `data/processed/`

## Faza 2 (CNN baseline)
- Jednostavan CNN (3 conv bloka + Dense)
- Trening i grafovi metrika
- Trenutni test accuracy: **0.173**
- Model se čuva lokalno u `models/` (ignorisan u git-u)

## Sledeće
- Ekstrakcija feature-a iz CNN-a + SVM klasifikator
- Confusion matrix i uporedna evaluacija
