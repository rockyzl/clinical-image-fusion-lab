# Interpretation boundary recorded before held-out access

Recorded during train-v2, before any analytical outer-validation or test label read.

The clinical-only linear models reached the prespecified100-epoch cap with best inner-tuning checkpoints at95/99/99. Their tuning losses were still falling slightly. They are **budget-capped**,not demonstrated to have converged. Retain all three results and show this limitation; do not claim that a fusion architecture is better than a fully optimized clinical-only baseline.

The primary contrast remains the originally specified **concat-linear minus image-linear**. Those controls have the same kind of linear classification head and differ by whether clinical features are added. Its validity requires checking both controls' complete curves before unlocking the holdout. MetaBlock and HEALNet are exploratory architecture comparisons,not evidence that complexity is intrinsically better.

Keep the fixed experimental budget rather than extending selected models after seeing test performance. Any future extended-budget experiment must be a separately declared lineage; test data is not available for tuning that follow-up. No winning architecture will be selected from test results for display: the browser will show all five models,seed17 fixed in advance.

Claims permitted: implementation, source adaptation, paired data handling, bounded training, grouped evaluation and research inference deployment. Claims excluded: clinical efficacy,clinical validation,publication-ready result,state of the art,convergence of capped baselines,or hospital deployment.
