"""Figure functions.  [IMPLEMENTED AT CHECKPOINT 5]

Every figure saved to figures/ as BOTH png (150 dpi) and pdf, consistent styling,
readable without color. Planned figures (see README "Visualizations"):
  1 learning curves (reward vs episode, median + IQR, QDQN vs MLP, 475 line)
  2 sample efficiency (same, x = env steps)
  3 scaling-parameter evolution (w[0], w[1], lam[i] vs step) -- visual proof of Failure Mode 1
  4 circuit diagram (qc.draw('mpl'))
  5 Q-value landscape heatmap Q(s,0)-Q(s,1), untrained vs trained
  6 loss & epsilon twin-axis diagnostic
  7 greedy rollout GIF
  8 param-count vs final performance scatter (headline figure)
"""
from __future__ import annotations
