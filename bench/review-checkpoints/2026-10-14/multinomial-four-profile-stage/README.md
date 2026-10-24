# Complete multinomial body in four storage profiles

All four complete source-body quotations passed Lean elaboration and explicit axiom gates with 35 unchanged recursive inputs. Scores storage is fp16, bf16, fp32 or fp64; every profile retains the actual cast to fp32 before cumsum, original i64 seed/offset/index/output storage, signed offset narrowing, both masked unspecified input loads, Bool reduction/argmax, the complete token loop and final masked store.

The macro varies only the Scores storage declaration. It retains all source binders, the two checked identity materializations and the common zeros/arange shape expression. This is complete-body frontend admission, not the ordinary paired corpus AST adapter, host wrapper coverage, full sampling execution/value theorem, or native refinement. The unchanged-source native matrix is in ../multinomial-native-original.
