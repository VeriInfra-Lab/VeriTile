import VeriTile.Triton.Float.CastExact.Idempotence

/-!
Exact cast laws for the concrete `.ieeeValue` software model. Finite same-format
casts preserve every word, including signed zero and subnormals. BF16 widening
and return preserves every non-NaN word; on all words its only change is the
model's explicit NaN canonicalization. The resulting ROUND-IDEM and
BF16-WIDEN-RETURN idempotence laws hold for all input words.
-/
