# Vector integer finite-word refinement

Three local modules refine the existing 81 canonical integer/bool pairs with
explicit finite source-width input words and exact output-word readback.
The public specification constructs complete ascending logical grid success
from two full tagged input-cell relations and a positive natural block size.
The independent initial-word result includes all modular promotion/add/store
steps and a complete memory frame. No new dtype pairs are counted.

Bool load normalization accepts all 256 raw bytes. The output is the entire
signed i8 storage byte: noncanonical output 3 is retained, and bool+bool wraps
at i1 before that store. Storage decode/re-encode roundtrips preserve every
word. This distinction is explicit in the finite-word reference.

Logical input/output region names may coincide. The two input relations then
require their decoded tagged cells to agree. The constructive memory theorem
states this exact compatibility; it introduces no source runtime guard.
Outside N, masked values are unconstrained. Native byte assembly, tags,
pointer validity, byte views/overlaps, finite addresses, allocator/compiler
errors and scheduling remain unproved structural correspondence.

No numerical callback or law is added: the inherited Algebra parameter is
unused in this integer graph. No closed-bit interpreter audit command or
whole-original-program acceptance is claimed. The frozen prior 81 TTIR
store-graph audit is reused by hash; no new frontend, native backend or GPU
run is performed. Source files and evidence require independent review before
repository integration.
