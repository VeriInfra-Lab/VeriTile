import bench.repairs.tritonbench_g.swiglu_triton.Proofs.DirectHost
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Words

namespace VeriTile.Bench.Repairs.SwigluTriton
open VeriTile.Triton
open Memory DirectPhysical DirectHost Native
noncomputable section

/-- All raw FP32 words and complete source graph. This optional bridge assumes the
six individual instruction ports in PrimitiveObligations; these are primitive
modeling boundaries, not task-level numerical experiments. FMA and sigmoid
lowering preserve their actual operand order and encoded constants. -/
theorem swiglu_direct_fp32 {M : Type} (ops : Ops M Word (BitVec 8))
    (profile : Profile) (native target : InstructionSemantics)
    (laws : PrimitiveObligations profile native target)
    (rt : Runtime backward) (entry : M) :
    HostSpec ops (compiledReference target)
      (fun p h plan s => NativeSteps ops native backward (layout p h) plan.order s)
      rt entry (execute ops (sourceScalar (compiledPorts native)) rt entry) := by
  apply host_spec
  · exact primitive_bridge profile native target laws
  · intro p h plan s
    apply native_steps ops native backward (layout p h) (valid p h) plan.order
    intro r hr
    exact List.mem_range.mp (plan.schedule.mem_iff.mp hr)

/-- Four actual returned bytes, using the standard little-endian FP32 storage
law. The value premise is the complete proved grid result, instantiated here. -/
theorem swiglu_direct_output_bytes {M : Type} (ops : Ops M Word (BitVec 8))
    (codec : Words.Codec ops) (profile : Profile) (native target : InstructionSemantics)
    (laws : PrimitiveObligations profile native target)
    (p : Prepared backward) (h : Admitted p) (plan : Plan p) (entry : M)
    (ch : Channel) (on : enabled backward ch) (row col : Nat) (hr : row < p.rows) (hc : col < p.cols)
    (byte : Nat) (small : byte < 4) :
    let final := DirectPhysical.run ops (sourceScalar (compiledPorts native)) backward (layout p h) plan.order entry
    let key := (p.tensor (outputRole ch)).key (row*p.cols+col)
    ops.observe final key.owner (key.address+byte) =
      (compiledReference target ch ((p.tensor .a).read ops entry (row*p.cols+col))
        ((p.tensor .b).read ops entry (row*p.cols+col))
        (if backward then (p.tensor .third).read ops entry (row*p.cols+col) else default)).extractLsb' (8*byte) 8 := by
  dsimp only
  rw [codec.byte _ _ byte small]
  have values := (run_spec ops _ _ (primitive_bridge profile native target laws) p h plan entry).1 ch on row col hr hc
  change ops.read _ _ = _ at values
  rw [values]

#axiomsClean swiglu_direct_fp32
#axiomsClean swiglu_direct_output_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton
