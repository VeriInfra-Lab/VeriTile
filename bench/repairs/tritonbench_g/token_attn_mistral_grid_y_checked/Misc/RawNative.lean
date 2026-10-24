import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Kernels
import bench.repairs.tritonbench_g.token_attn_mistral.Native

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.RawNative
open VeriTile.Triton FP.Structural TokenAttnMistral
noncomputable section

/-- Complete source execution on raw word/metadata cells, with the original
primitive instructions, ordered reductions and legal adjacent contractions.
The native result includes padded operations and the full tagged-memory frame. -/
theorem kernel_native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (p : Parameters) (work : List Grid.PID) (memory : Grid.Memory Native.Word)
    (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    ∃ final, Grid.run Native.algebra (Native.reduction R) p work (Native.entry memory) = some final ∧
      (∀ e ∈ Grid.events p work,
        Native.observe I S final p.output (Grid.offset p e.1 e.2) =
          Native.eval I S (Grid.values Native.algebra (Native.reduction R) p (Native.entry memory) e)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ e ∈ Grid.events p work, o ≠ Grid.offset p e.1 e.2) →
        final r o = Native.entry memory r o) :=
  Native.complete I R S p work memory separate unique

#axiomsClean kernel_native
#specNonCircular Grid.values avoiding [kernel,FP.Structural.exec,Grid.run]
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.RawNative
