# Complete integer and mixed program permutations

Three modules and five public specifications refine the existing 135 integer/bool
and mixed fp32/fp16/bf16 profiles. Alongside the prior 25 floating profiles, all
160 currently represented value/grid pairs admit arbitrary complete whole-program
permutations. There are zero new dtype pairs and zero accepted original programs.

The generic schedule argument is instantiated with actual source execution and
same-offset locality proofs. Public premises contain no execution-success or
output-correctness assumption. Programs read actual preceding memory. References
use initial memory and preserve the complete outside-output frame. Equal logical
region names are allowed; physical byte aliases and native interleavings remain
unproved. Total undefined conversion policies and active defined-domain results
keep their prior numerical boundaries. This package adds no numerical rules.

All three final compilations passed: 16 axiom checks, 2 noncircular checks, 3 module
audits, no warnings. Exact dependency fingerprints and an isolated staged
whitespace check accompany the receipt. No new native/compiler/GPU run is claimed.
