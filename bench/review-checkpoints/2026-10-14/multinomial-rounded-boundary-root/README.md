# Multinomial rounded boundary obligation

Six Lean theorems and their axiom gates passed with nine unchanged recursive inputs. They use the existing exact-rational IEEE-value software operations and ordinary `decide +kernel`; no native-decide axiom or GPU execution is used.

Canonical Philox seed 0, offset 1, 32-bit offsets and ten rounds produce fp32 bits `0x3d63666a`. Let that value be the previous cumulative value and let the next score be `0x3e800000` (0.25). Rounded addition produces `0x3e9c6ccd`; subtracting the score produces `0x3d636668`, strictly smaller than the previous cumulative value. The actual sample lies strictly between this computed lower endpoint and the new cumulative value.

Thus a future proof cannot replace the source expression `cum_scores - scores` with the preceding accumulator without numerical hypotheses. This scalar boundary witness is not yet a full multinomial execution counterexample, a normalized-distribution theorem, or hardware conformance evidence. Its role is to guide the exact value contract and preserve the source's last-matching-block behavior.
