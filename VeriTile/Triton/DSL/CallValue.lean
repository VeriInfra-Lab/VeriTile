import VeriTile.Triton.Core

/-! Internal source-call provenance shared by type propagation and expression
expansion. Its existing qualified name is retained for frontend compatibility.
The call elaborator checks actual operation precision before removing it. -/
namespace VeriTile.Triton.DSL.TypedStores
/-- Internal provenance marker for a supplied FP32 literal. The call frontend
checks every occurrence's actual computation precision after elaboration. -/
@[irreducible] def fp32CallValue (value : ℝ) : ℝ := value

end VeriTile.Triton.DSL.TypedStores
