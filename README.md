# Motor Imagery Transfer Learning Study

This study aims to investigate the possibility make short-time motor imagery calibration using transfer learning. Euclidean Alignment (EA)[1] and Weight Logistic Regression-Based Learning (wLTL)[2] will be used for the 4 classes of motor imagery transfer learning.     

**Target Domain Dataset**
1. Unicorn Hybrid Black: The motor imagery data from 7 participants recorded from my [undergraduate thesis](https://suparach3.wordpress.com/blog/), contain 120 trials per subject, 40 trials per class (Left hand, Right hand, Both Feet and Non-imagine).

**Source Domain Dataset**
1. [Physionet](https://archive.physionet.org/pn4/eegmmidb/): The Motor Imagery data from 109 participants of Left hand, Right hand, Both feet and Rest as Non-imagine from this dataset, contain 25 trials per class for each subject (counter-balance for Non-imagine class).
2. [BCI competition IV 2a](https://www.bbci.de/competition/iv/): Contain 4 classes of Left hand, Right hand, Feet and Tongue motor imagery, 9 participants tasked to perform motor imagery 72 trials per class (all sessions) 

I selected EEG channels Fz, C3, Cz, C4, and Pz for transfer learning implementation for all datasets. 

# Inter-dataset Result
I used Unicorn Hybrid Black dataset, separated into the source and target domains. In this result, every subject will be selected as the target domain, otherwise the source domain. For the target domain dataset, some trials will be selected as calibration sets in the EA and wLTL approach.  

