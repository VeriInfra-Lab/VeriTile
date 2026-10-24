import bench.repairs.tritonbench_g.attention_score_tile_checked.Kernels
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.CurrentMachine

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.RawNative
open VeriTile Triton Reused Geometry Physical NativeInstructions
noncomputable section

/-- Independent source-ordered scalar/fixed-MMA graph over raw input words.
Loaded words retain their bits; the supplied ISA interprets native primitives. -/
def kernelNativeTarget (I : ISA (BitVec 32)) (c : Shape) (abi : ABI c)
    (physical : Nat → BitVec 32) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) : BitVec 32 :=
  (NativePhysical.target c abi physical offset size scale window complement z h j).eval
    { I with bits := id }

/-- Complete typed-source execution and native output observations, with
preservation of every input region and every exterior word. Shape, scalar,
allocation and machine-range assumptions contain no Tensor API metadata. -/
theorem kernel_native (I : ISA (BitVec 32)) (c : Shape) (abi : ABI c)
    (physical : Nat → BitVec 32) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (domain : Whole002.Representation.Domain c abi offset size window) (tile : c.B*c.D≤1048576) :
    Whole002.Machine.Closed c abi offset size window ∧
      (∀ shape ∈ statementListShapes (kernel c offset size scale window complement).surfaceBody,
        shape.prod≤1048576) ∧
      ∃ result, nativeExecute c abi physical offset size scale window complement=some result ∧
        (∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
          (result ((abi.alloc "out").base+c.outputBase z.val h.val+j.val)).eval { I with bits := id }=
            kernelNativeTarget I c abi physical offset size scale window complement z h j) ∧
        (∀ (r : RegionName), tracked r → r≠"out" → ∀ i, i<(abi.alloc r).length →
          result ((abi.alloc r).base+i)=Graph.bits (physical ((abi.alloc r).base+i))) ∧
        (∀ a, ¬(abi.alloc "out").Contains a → result a=Graph.bits (physical a)) := by
  obtain ⟨result,run,values,inputs,frame⟩ := NativePhysical.returned c abi physical
    offset size scale window complement
  refine ⟨Whole002.Machine.closed c abi offset size window domain,
    kernel_tiles_bounded c offset size scale window complement tile,result,?_,?_,inputs,frame⟩
  · simpa only [nativeExecute,Whole002.Execution.Native.execute_eq] using run
  · intro z h j
    exact congrArg (Graph.eval { I with bits := id }) (values z h j)

#specNonCircular kernelNativeTarget avoiding [nativeExecute,FP.Structural.exec,NativePhysical.execute]
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.RawNative
