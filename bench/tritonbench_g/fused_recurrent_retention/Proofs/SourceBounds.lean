import bench.tritonbench_g.fused_recurrent_retention.AddressSource
import bench.tritonbench_g.fused_recurrent_retention.Proofs.API
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section

namespace SourceAddress
structure Bounds (phase : SourcePhase) (seed extra : Bool) (s : Shape) (b : InputBuffers) : Prop where
  args : ∀ n∈[s.B,s.H,s.T,s.DK,s.DV,s.BK,s.BV,s.NK,s.NV,s.BH,s.NV*s.NK,s.NV*s.NK*s.BH],signed32 n
  qvArgs : ∀ i : Fin 3,signed32 (b.q.strides ⟨i.val+1,by omega⟩) ∧ signed32 (b.v.strides ⟨i.val+1,by omega⟩)
  cotangentArgs : phase=.backward → ∀ i,signed32 (b.do_.strides i) ∧
    (extra=Bool.true → signed32 (b.finalGrad.strides i))
  grid : s.NV<2^31 ∧ s.NK≤65535 ∧ s.BH≤65535
  expressions : ∀ iv<s.NV,∀ ik<s.NK,∀ h<s.BH,∀ lk<s.BK,∀ lv<s.BV,
    ∀ e∈roots phase seed extra s b iv ik h lk lv,e.Safe
  pointers : ∀ iv<s.NV,∀ ik<s.NK,∀ h<s.BH,∀ lk<s.BK,∀ lv<s.BV,∀ r ≤ s.T,
    ∀ offset∈formed phase seed extra s b iv ik h lk lv r,signed64 (4*offset)

theorem exact (phase : SourcePhase) (seed extra : Bool) (s : Shape) (b : InputBuffers)
    (bounds : Bounds phase seed extra s b) (iv ik h lk lv : Nat)
    (hv : iv<s.NV) (hk : ik<s.NK) (hh : h<s.BH) (hlk : lk<s.BK) (hlv : lv<s.BV)
    (e : IExpr) (he : e∈roots phase seed extra s b iv ik h lk lv) : e.machine.toInt=e.value :=
  e.machine_exact (bounds.expressions iv hv ik hk h hh lk hlk lv hlv e he)
end SourceAddress

#axiomsClean SourceAddress.exact
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
