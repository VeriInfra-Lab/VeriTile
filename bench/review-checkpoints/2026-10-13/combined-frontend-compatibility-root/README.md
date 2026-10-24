# Combined frontend compatibility regressions

All 17 targeted Lean files passed with 179 unchanged recursive local inputs.
The sources directory is an exact hash-checked snapshot of those inputs.
This includes `tl.advance` storage provenance and the restricted legacy
pointer-element cast compatibility path, alongside the finite helper,
branch dtype and block-pointer regressions.

These are local Lean and declared axiom-gate results. The aggregate build
is recorded separately; no native execution or independent comparator
acceptance follows from this checkpoint.
