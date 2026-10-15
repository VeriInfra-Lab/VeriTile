# Constant-branch FP rewriting — passed

All six branch, range and fp32-addition regression targets compiled with exit code 0. The 92 captured inputs were unchanged. The new lemmas preserve arbitrary unreachable arms and still require the selected arm guard. Negative checks reject missing guards, the wrong branch, the wrong prefix and absent arithmetic rules. These are local Lean and axiom checks, not independent comparator replay.
