
def classify_risk(rul):
    if rul > 80:
        return 0   # Low
    elif rul > 40:
        return 1   # Medium
    else:
        return 2   # High
