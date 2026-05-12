# Boosting Mistake Lab

Boosting Mistake Lab is an interactive Level 2 demo from **Interactive ML Labs**.

The demo will show how boosting builds a strong classifier from multiple weak learners trained sequentially.

## What this demo will teach

This demo focuses on boosting intuition.

The key idea:

```text
start with equal sample weights
-> train weak learner
-> find mistakes
-> increase weights of misclassified samples
-> train next weak learner
-> combine weak learners
```