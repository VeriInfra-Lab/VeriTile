import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Proofs.NativeBinding
import bench.repairs.tritonbench_g.chunk_delta_fwd.GridOutputs
import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.ScalarGraph

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.RawNative
open VeriTile Triton FP.Structural ChunkDeltaFwd
open VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph (WordPrimitives)
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging (Expr)
noncomputable section

/-- Full raw source execution and all native output words of the independent
input-reader recurrence, with arbitrary initial exterior memory preserved. -/
theorem chunk_delta_kernel_native (I : WordPrimitives) (p : Parameters)
    (shape : Geometry.ShapeValid p) (bits : RegionName → Nat → BitVec 32)
    (before : Memory Expr) (inputs : SameInputs before (Native.inputs bits))
    (schedule : List PID) (order : schedule.Perm (grid p)) :
    ∃ final,runGrid Native.algebra Native.dot p schedule before=some final ∧
      (∀ head,head<p.B*p.H → ∀ chunk,chunk<p.NT → ∀ key (hk : key<p.K),∀ value,value<p.V →
        (Native.lower ((final "history" (((head*p.NT+chunk)*p.K+key)*p.V+value)).read .real)).eval I.isa=
          (NativeReference.history p shape (NativeReference.inputs p bits) head chunk key value hk).eval I.isa) ∧
      (∀ head,head<p.B*p.H → ∀ time,time<p.T → ∀ value,value<p.V →
        (Native.lower ((final "corrected" ((head*p.T+time)*p.V+value)).read .real)).eval I.isa=
          (NativeReference.corrected p shape (NativeReference.inputs p bits) head time value).eval I.isa) ∧
      (p.storeFinal=Bool.true → ∀ head,head<p.B*p.H → ∀ key (hk : key<p.K),∀ value,value<p.V →
        (Native.lower ((final "final" ((head*p.K+key)*p.V+value)).read .real)).eval I.isa=
          (NativeReference.history p shape (NativeReference.inputs p bits) head p.NT key value hk).eval I.isa) ∧
      (∀ r o,¬OutputCell p r o → final r o=before r o) := by
  obtain ⟨final,ran,result⟩ := whole_grid Native.algebra Native.dot p shape schedule order
    (Native.inputs bits) before inputs
  refine ⟨final,ran,?_,?_,?_,result.frame⟩
  · intro head hh chunk hc key hk value hv
    rw [result.history head hh chunk hc key hk value hv,Cell.read_same]
    exact congrArg (fun graph => graph.eval I.isa) (NativeReference.history_binding p shape bits head chunk key value hk)
  · intro head hh time ht value hv
    rw [result.corrected head hh time ht value hv,Cell.read_same]
    exact congrArg (fun graph => graph.eval I.isa) (NativeReference.corrected_binding p shape bits head time value)
  · intro on head hh key hk value hv
    rw [result.final on head hh key hk value hv,Cell.read_same]
    exact congrArg (fun graph => graph.eval I.isa) (NativeReference.history_binding p shape bits head p.NT key value hk)

#axiomsClean chunk_delta_kernel_native
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.RawNative
