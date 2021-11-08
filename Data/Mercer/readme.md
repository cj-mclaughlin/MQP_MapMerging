# Original Notation
occupied -> 0
unknown -> 127
free -> 255

In practice, we can rearrange values s.t. unknown = 0.5 and free = 1 (and thus each cell represents the probability that we believe the cell to be free)
