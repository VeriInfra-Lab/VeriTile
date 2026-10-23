import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.DirectHost
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Words

namespace VeriTile.Bench.Repairs.GegluTanhTriton
open VeriTile.Triton
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical DirectHost Native
noncomputable section

/-- Optional historical host result with an explicit local model-conformance
interface. The instruction ports are not pending numerical experiments for
this task; current public entries prove kernels with one instruction model. -/
specification geglu_direct_fp32 {M : Type} (ops : Ops M Word (BitVec 8))
    (native target : Instructions) (laws : PrimitiveObligations native target)
    (rt : Runtime backward) (entry : M) :
    HostSpec ops (compiledReference target)
      (fun p h plan s => Native.Certificates ops native backward (layout p h) plan.order s)
      rt entry (execute ops (sourceScalar (compiledPorts native)) rt entry) := by
  apply host_spec
  · exact primitive_bridge native target laws
  · intro p h plan s
    exact Native.certificates ops native backward (layout p h) (valid p h) plan.order s

theorem geglu_direct_output_bytes {M : Type} (ops : Ops M Word (BitVec 8))
    (codec : SwigluTriton.Words.Codec ops) (native target : Instructions)
    (laws : PrimitiveObligations native target)
    (p : Prepared backward) (h : Admitted p) (plan : Plan p) (entry : M)
    (ch : Channel) (on : enabled backward ch) (i : Nat) (hi : i < p.rows*p.cols)
    (byte : Nat) (small : byte < 4) :
    let final := Physical.run ops (sourceScalar (compiledPorts native)) backward (layout p h) plan.order entry
    let key := (p.tensor (outputRole ch)).key i
    ops.observe final key.owner (key.address+byte) =
      (compiledReference target ch ((p.tensor .a).read ops entry i) ((p.tensor .b).read ops entry i)
        (if backward then (p.tensor .third).read ops entry i else default)).extractLsb' (8*byte) 8 := by
  dsimp only
  rw [codec.byte _ _ byte small]
  have values := (run_spec ops _ _ (primitive_bridge native target laws) p h plan entry).1 ch on i hi
  change ops.read _ _ = _ at values
  rw [values]

#axiomsClean geglu_direct_fp32
#axiomsClean geglu_direct_output_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton
