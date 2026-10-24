# Signed offset narrowing establishes RNG validity

Three Lean theorems with explicit axiom gates prove that the actual signed i32 conversion establishes the offset representation domain for arbitrary mathematical Int inputs. The tensor theorem covers every shape, including empty shapes. The typed execution theorem composes successful seed/offset operand evaluation, the actual Op.intToSigned expression and the canonical vector-seeded RNG result for arbitrary Structural.Algebra, ambient precision and round count.

The remaining explicit range premise concerns the i64 seed values. There is no bound premise on the original offset values and no assumed successful execution of the complete RNG expression. This is an expression-level execution/range proof supporting multinomial admission, not a complete sampling kernel or native refinement theorem.
