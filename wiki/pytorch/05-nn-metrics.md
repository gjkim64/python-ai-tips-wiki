## DL performance indicators

## ROC AUC 

What the Saturating Shape Means

• Early steep rise. As you slowly lower your cutoff threshold from the far-left edge (0,0), the curve jumps straight up toward the top-left corner. This shows the model finds almost all positive cases first before accidentally flagging negative ones.
• Flat top near 1.0. Once the curve hits a TPR near 1, it flattens out across the top as the False Positive Rate (FPR) increases. A flat ceiling means you have already caught nearly every positive case, so relaxing your rules further only adds false alarms without finding new positives.

## PR AUC 

• Precision (Quality of alerts). Out of every item the model flagged as positive, how many were actually positive? High precision means very few false alarms.
• Recall (Quantity found). Out of all the real positive cases in your data, how many did the model successfully catch? High recall means very few missed cases.

When positive cases are extremely rare—like detecting fraud in 0.1% of transactions—ROC AUC can look artificially high because there are millions of easy negative cases to get right. PR-AUC solves this blind spot by judging the model solely on its performance with the tiny positive needle in the haystack.

• a great PR curve pushes toward the top-right corner, where both precision and recall stay close to 1.0 at the same time.

Note ROC AUC ideal is toward top left corner

## Classification
- simple log loss
- simple accuracy alone - misleading when certain class is rare in counts
- simple precision (proportion of positive predictions that are correct, note there is correct negative prediction too)
- recall - proportion of positive cases the model identifies (count)
- F1 score - balanced score of precision and recall (something like PR AUC?)
- MCC - TP TN FP FN all mixed up

## Regression metrics
- MAE - means absolute error
- MSE - Mean squred error (penalize large errors heavily)
- R2 - correlation, variance

## Specialized metrics
- IoU and mAP - object detection
- etc.

