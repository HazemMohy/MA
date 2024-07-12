import numpy as np

def calc_volumetric_score(tp, tn, fp, fn):
    """Calculate basic volumetric scores based on a confusion matrix
    Args:
        - tp        (int)           : True poitives
        - tn        (int)           : True negatives
        - fp        (int)           : False positives
        - fn        (int)           : False negatives
    Returns:
        - precision (float)         : Precision
        - recall    (float)         : Recall/Sensitivity
        - vs        (float)         : Volumetric Similarity
        - f1_dice   (float)         : F1/Dice Score
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    vs = 1 - abs(fn - fp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    if tp == 0 and tn == 0 and fp == 0 and fn == 0:
        f1_dice = 1
    else:
        f1_dice = (2*tp) / (2*tp + fp +fn)
    return precision, recall, vs, accuracy, f1_dice

def confusion_matrix(pred, target):
    """Calculate test metrices
    Args:
        - pred      (torch.tensor)  : The predicted values in binary (0,1) format
        - target    (torch.tensor)  : The target value in a binary(0,1) format
    Returns:
        - tp        (int)           : True poitives
        - tn        (int)           : True negatives
        - fp        (int)           : False positives
        - fn        (int)           : False negatives
    """
    # pred = np.asarray(pred).astype(bool)
    # target = np.asarray(target).astype(bool)

    la = lambda a, b: np.logical_and(a, b)
    no = lambda a: np.logical_not(a)

    tp = la(pred, target).sum()
    tn = la(no(pred), no(target)).sum()
    fp = la(pred, no(target)).sum()
    fn = la(no(pred), target).sum()
    return tp, tn, fp, fn
