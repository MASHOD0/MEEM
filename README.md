# MEEM: Robust Tracking via Multiple Experts Using Entropy Minimization
a multi-expert restoration scheme to address the model drift problem in online tracking.
- a tracker and its historical snapshots constitute an expert ensemble, where the best expert is selected to restore the current tracker when needed based on a minimum entropy criterion,
so as to correct undesirable model updates.
- the base tracker in our formulation exploits an online SVM on a budget algorithm and an explicit feature mapping
method for efficient model update and inference.

the proposed multi-expert restoration scheme significantly improves
the robustness of our base tracker, especially in scenarios with frequent occlusions and repetitive appearance variations.