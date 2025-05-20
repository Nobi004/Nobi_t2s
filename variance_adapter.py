"""Variance Adaptor
Adds duration, pitch, and energy to the hidden representation to address the one-to-many 
mapping problem (multiple speech variations for one text).
Includes predictors for each variance and projects them to the hidden dimension"""

import torch