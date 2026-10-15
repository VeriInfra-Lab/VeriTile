# Final signed-cast frontend context regressions

All six requested Lean files passed with 68 unchanged recursive inputs. This final stage includes the complete fp32 multinomial body, signed/unsigned cast and random regressions, seven evaluator laws, arbitrary-input and boundary execution theorems, and nested direct/parenthesized/functional RNG, condition and address-use checks.

Signed i32 results retain low bits with signed interpretation; Bool-to-i32 produces 0/1. Nested load annotations preserve the declared source integer channel before narrowing, and address inference retains the signed cast result until the final address conversion. Explicit random provenance permits i64-to-i32 only with these actual expression paths. Unsupported other signed profile conversions remain rejected.

This supersedes the initial isolated cast receipt for nested/address contexts. It is local Lean regression evidence, not ordinary corpus source-pair acceptance, a complete multinomial execution/value theorem, native refinement or an independent comparator replay.
