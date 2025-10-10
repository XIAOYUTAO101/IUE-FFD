To assess the sensitivity of our temporal partitioning, beyond the time windows reported in the main text we further conduct robustness checks using additional windows: 60 and 120 days for the CCT dataset. Accordingly, we report the F1-Macro metric and highlight a subset of key baselines. Among them, Table 2 and Table 4 correspond to a 60-day window with a total of 12 slot stages, while Table 1 and Table 3 correspond to a 120-day window with a total of 6 slot stages.

Table1.

|    Slot     | PCGNN | GHRN  | DGA-GNN |  PMP  | IUE-GMP* |
| :---------: | :---: | :---: | :-----: | :---: | :------: |
|    slot1    | 81.38 | 88.51 |  92.57  | 93.05 |  94.38   |
| slot1-slot2 | 79.52 | 87.35 |  90.37  | 91.01 |  93.66   |
| slot1-slot3 | 77.45 | 85.62 |  91.43  | 90.87 |  92.10   |
| slot1-slot4 | 76.39 | 84.38 |  89.56  | 89.55 |  91.34   |
| slot1-slot5 | 74.33 | 82.55 |  87.93  | 88.34 |  90.75   |
| slot1-slot6 | 75.06 | 81.36 |  85.60  | 87.11 |  89.88   |

Table2.

|     Slot     | PCGNN | GHRN  | DGA-GNN |  PMP  | IUE-GMP* |
| :----------: | :---: | :---: | :-----: | :---: | :------: |
|    slot1     | 80.83 | 85.69 |  90.55  | 91.07 |  92.35   |
| slot1-slot2  | 79.69 | 85.45 |  89.50  | 92.38 |  94.38   |
| slot1-slot3  | 79.83 | 85.26 |  90.89  | 92.76 |  94.10   |
| slot1-slot4  | 80.52 | 86.15 |  90.10  | 91.85 |  93.54   |
| slot1-slot5  | 79.34 | 84.39 |  89.42  | 90.42 |  92.32   |
| slot1-slot6  | 78.81 | 83.51 |  88.72  | 91.16 |  91.56   |
| slot1-slot7  | 78.47 | 82.16 |  87.66  | 89.42 |  91.61   |
| slot1-slot8  | 76.34 | 81.35 |  86.16  | 88.53 |  91.83   |
| slot1-slot9  | 76.47 | 81.60 |  85.38  | 87.42 |  89.65   |
| slot1-slot10 | 75.34 | 80.09 |  85.86  | 87.36 |  89.32   |
| slot1-slot11 | 74.39 | 79.56 |  84.36  | 86.51 |  89.51   |
| slot1-slot12 | 73.22 | 79.22 |  83.51  | 85.47 |  88.62   |

Table3.

|    Slot     |  EWC  |  MAS  | PI-GNN |  CCL  | IUE-GMP |
| :---------: | :---: | :---: | :----: | :---: | :-----: |
|    slot1    | 94.10 | 93.96 | 83.75  | 85.10 |  94.15  |
| slot1-slot2 | 93.25 | 92.54 | 82.37  | 84.35 |  94.01  |
| slot1-slot3 | 93.01 | 92.37 | 80.52  | 83.09 |  93.57  |
| slot1-slot4 | 92.42 | 91.66 | 79.68  | 82.77 |  93.05  |
| slot1-slot5 | 91.68 | 90.19 | 78.35  | 80.46 |  92.76  |
| slot1-slot6 | 90.23 | 89.53 | 77.47  | 79.67 |  91.36  |

Table4.


|     Slot     |  EWC  |  MAS  | PI-GNN |  CCL  | IUE-GMP |
| :----------: | :---: | :---: | :----: | :---: | :-----: |
|    slot1     | 91.37 | 92.10 | 81.52  | 83.62 |  92.35  |
| slot1-slot2  | 93.05 | 93.55 | 82.39  | 82.55 |  94.10  |
| slot1-slot3  | 92.35 | 93.21 | 81.71  | 82.30 |  94.62  |
| slot1-slot4  | 93.57 | 92.85 | 80.86  | 81.16 |  94.07  |
| slot1-slot5  | 92.61 | 92.73 | 79.37  | 82.04 |  93.59  |
| slot1-slot6  | 91.03 | 91.59 | 78.53  | 80.52 |  93.63  |
| slot1-slot7  | 91.80 | 91.77 | 77.74  | 79.50 |  92.51  |
| slot1-slot8  | 90.34 | 90.82 | 78.10  | 78.14 |  92.10  |
| slot1-slot9  | 90.20 | 90.49 | 77.67  | 78.47 |  91.33  |
| slot1-slot10 | 89.63 | 91.20 | 75.35  | 77.36 |  91.42  |
| slot1-slot11 | 89.35 | 90.05 | 74.94  | 76.51 |  90.63  |
| slot1-slot12 | 88.02 | 88.37 | 73.61  | 75.85 |  89.50  |



In our ablation analyses, we integrated the IKT component into representative baselines (e.g., PMP,DGA-GNN and GHRN) to further assess its contribution.

| Model Variant  | F1-Macro | Recall |  FPR  |  AUC  |
| :------------: | :------: | :----: | :---: | :---: |
|  GHRN w/ IKT   |  83.34   | 84.67  | 15.33 | 93.74 |
| DGA-GNN w/ IKT |  86.51   | 88.14  | 11.86 | 94.36 |
|   PMP w/ IKT   |  88.60   | 89.68  | 10.32 | 97.10 |



We further broadened our evaluation by adding Yelp and T-Finance as supplementary experiments. We note that most public datasets are constructed in a closed manner, which only partially aligns with our open-world objective. To better approximate the target setting, we partition each dataset into chronological subgraphs (“slots”) based on (simulated) timestamps and perform slot-wise training and evaluation, thereby simulating an incremental learning protocol. Both datasets are divided into five slots, we report the F1-Macro metric and highlight a subset of key baselines. Table 5 and Table 6 present the results on the Yelp dataset, while Table 7 and Table 8 show the results on the T-Finance dataset.

Table5.


|    Slot     | PCGNN | GHRN  | DGA-GNN |  PMP  | IUE-GMP* |
| :---------: | :---: | :---: | :-----: | :---: | :------: |
|    slot1    | 65.18 | 77.43 |  80.54  | 81.57 |  82.60   |
| slot1-slot2 | 63.32 | 76.26 |  79.60  | 80.72 |  81.64   |
| slot1-slot3 | 62.75 | 74.88 |  78.12  | 80.65 |  80.56   |
| slot1-slot4 | 61.60 | 73.25 |  77.04  | 79.18 |  80.12   |
| slot1-slot5 | 60.42 | 72.31 |  76.26  | 78.60 |  79.59   |

Table6.


|    Slot     |  EWC  |  MAS  | PI-GNN |  CCL  | IUE-GMP |
| :---------: | :---: | :---: | :----: | :---: | :-----: |
|    slot1    | 81.95 | 82.20 | 60.60  | 62.05 |  82.35  |
| slot1-slot2 | 81.20 | 81.46 | 59.33  | 61.38 |  81.28  |
| slot1-slot3 | 80.57 | 80.55 | 58.51  | 60.63 |  81.67  |
| slot1-slot4 | 79.50 | 78.79 | 58.30  | 59.48 |  80.12  |
| slot1-slot5 | 78.61 | 77.54 | 56.13  | 58.55 |  80.51  |

Table7.


|    Slot      | PCGNN  |  GHRN  | DGA-GNN |  PMP   | IUE-GMP* |
| :---------: | :----: | :----: | :-----: | :----: | :------: |
|    slot1    | 59.64 | 85.51 | 89.13 | 88.89 |  89.56  |
| slot1-slot2 | 58.28 | 86.37 | 89.41 | 89.79 |  90.35  |
| slot1-slot3 | 57.06 | 87.20 | 88.09 | 90.01 |  91.46  |
| slot1-slot4 | 55.45 | 86.59 | 88.34 | 89.25 |  91.63  |
| slot1-slot5  | 54.67 | 86.10 | 87.25 | 89.10 |  90.39  |

Table8.

|    Slot     |  EWC  |  MAS  | PI-GNN |  CCL  | IUE-GMP |
| :---------: | :---: | :---: | :----: | :---: | :-----: |
|    slot1    | 88.68 | 88.64 | 56.13  | 60.19 |  89.69  |
| slot1-slot2 | 89.56 | 89.96 | 57.24  | 61.83 |  90.73  |
| slot1-slot3 | 90.57 | 91.34 | 57.19  | 62.74 |  92.62  |
| slot1-slot4 | 89.15 | 90.52 | 55.52  | 61.37 |  91.86  |
| slot1-slot5 | 89.62 | 90.16 | 54.39  | 60.09 |  91.02  |