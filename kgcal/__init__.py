"""Select-and-repair pruning of SwiGLU feed-forward units with knowledge-graph calibration text.

Modules: data (PopQA split, Wikidata statements, verbalizers, calibration texts, C4), model_ops (loading, feed-forward
statistics, GRIDE widths), compress (selection rules, ridge repair, Lemma 1 terms), evaluate (fact recall, perplexity,
zero-shot accuracy), run_grid (one model, its budgets and a grid of calibration conditions), estimators (the
intrinsic-dimension estimators of the width check) and paths (every input and output location).
"""
