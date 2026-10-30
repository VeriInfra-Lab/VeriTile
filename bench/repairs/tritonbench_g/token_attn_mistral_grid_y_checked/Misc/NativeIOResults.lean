import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Proofs.PublicIO
import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Misc.NativeReference

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.PublicIO
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat)
    (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    (IO.native R p work counts).ObservesOn (IO.observe I S p work)
      (Ports.Domain p work counts) (Reference.native I R S p work counts) := by
  have h := ProgramIO.ObservesOn.of_implementsOn
    (implements Native.algebra (Native.reduction R) p work counts separate unique) (IO.observe I S p work)
  convert h using 1
  funext vals o i
  have ho : o=⟨0,by simp [IO.native,IO.program,Ports.memory,Ports.outputs]⟩ := by
    apply Fin.ext
    change o.val=0
    have h : o.val<1 := o.isLt
    omega
  subst o
  rfl

#axiomsClean native
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.PublicIO
