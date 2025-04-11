1. Contains updated memorization code, has different metrics to assess why certain nodes are special. None of them conclusive yet.
2. Also contains finetuning.py which runs logistic,ridge and MLP by using representations from f_model.pt, drops nodes and measures test accuracy to assess what happens when we drop top-k memorized nodes vs random non-memorized ones.
