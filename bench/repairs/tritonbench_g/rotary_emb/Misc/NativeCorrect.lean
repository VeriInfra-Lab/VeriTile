import bench.repairs.tritonbench_g.rotary_emb.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.RotaryEmb.NativeCorrect
open VeriTile.Triton FP.Structural
noncomputable section

/-- The FP32 source yields the independent selected mul/neg/fma graph and is
equivalent to the existing integer-address candidate, on every native node
and FP32 word. Both actual sources execute safely and preserve the exterior
cell frame. `p` supplies primitive instruction meanings, with no additional
floating rewrite law or finite-payload restriction. -/
theorem rotary_emb_native_equiv (rest : Algebra Native.Node) (p : Native.Instructions)
    (shape : Shape) (valid : shape.Valid) (pids : Nat → Nat) :
    (RotaryIO.kernelIO rest (Native.ops p) shape pids).Implements (RotaryIO.nativeRotation p shape) ∧
      (RotaryIO.kernelIO rest (Native.ops p) shape pids).Equivalent
        (RotaryIO.candidate rest (Native.ops p) shape pids) :=
  RotaryIO.native_result rest p shape valid pids

#stmtSurfaceSubset rotary_emb_native_equiv ⊆
  [FP.Structural.Algebra,Native.Node,Typed.instInhabitedNode,Native.Instructions,Shape,Shape.Valid,RotaryIO.kernelIO,
   Native.ops,RotaryIO.nativeRotation,RotaryIO.candidate,ProgramIO.Implements,ProgramIO.Equivalent]
#specNonCircular RotaryIO.nativeRotation avoiding
  [Kernels.kernel,Candidate.optimized,Typed.execution,FP.Structural.exec]
#axiomsClean rotary_emb_native_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryEmb.NativeCorrect
