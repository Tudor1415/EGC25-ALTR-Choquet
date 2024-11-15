import math

def entropy(nx0, nx1, n):
    """Calculates entropy."""
    if nx0 == 0 or nx1 == 0:
        return 0

    positive = 0
    if nx1 > 0:
        positive = (nx1 / n) * math.log(nx1 / n, 2)  # Log base 2 for entropy

    negative = 0
    if nx0 > 0:
        negative = (nx0 / n) * math.log(nx0 / n, 2)  # Log base 2 for entropy

    return -(positive + negative)

def information_gain(nx0, nx1, n11, n00, n01, n0x, n):
    """Calculates information gain."""
    H_D = entropy(nx0, nx1, n)
    freq_D_F = n11 / n
    H_D_F = entropy(0, n11, n11)
    freq_bar_D_F = (n - n11) / n
    H_D_bar_F = entropy(n00, n01, n0x)

    value = H_D - freq_D_F * H_D_F - freq_bar_D_F * H_D_bar_F
    return value

def phi(n, n11, n1x, nx1, n0x, nx0):
    """Calculates the phi measure (φ) from a 2x2 contingency table."""
    
    numerator = n * n11 - n1x * nx1
    denominator = math.sqrt(n1x * nx1 * n0x * nx0)
    
    if denominator == 0: 
        return 0.0
    
    return numerator / denominator