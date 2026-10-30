import bench.repairs.tritonbench_g.attention_score.Proofs.CurrentExecution
import bench.repairs.tritonbench_g.attention_score.Proofs.CurrentMachine

/-! Public native FP32 path. Actual-source lowering preserves the TF32 MMA
seed and feature order, scale-first FMA, sign negation, ex2.approx.ftz, ordered
B16/32/64 reduction, cast, full grid and physical return. Numerical obligations
are explicit scalar/fixed-ISA ports; none is a whole dot, reduction or output.
The compiler/runtime boundary is ordinary operand/layout/control mapping. -/
namespace VeriTile.Bench.Repairs.AttentionScore.Whole002.FPEquiv
open VeriTile Triton NativeInstructions Geometry Physical
noncomputable section

theorem native_return (c : Shape) (dispatch : Dispatch c) (abi : ABI c) (physical : Nat → BitVec 32)
    (offset size : Int) (hostScale : ℝ) (window complement : Bool)
    (domain : Representation.Domain c abi offset size window) :
    Machine.Closed c abi offset size window ∧ persistentBlock dispatch=c.B ∧
    ∃ result, Execution.Native.execute c abi physical offset size hostScale window complement=some result ∧
      (∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
        result ((abi.alloc "out").base+c.outputBase z.val h.val+j.val)=
          NativePhysical.target c abi physical offset size hostScale window complement z h j) ∧
      (∀ (r : RegionName), tracked r → r≠"out" → ∀ i, i<(abi.alloc r).length →
        result ((abi.alloc r).base+i)=Graph.bits (physical ((abi.alloc r).base+i))) ∧
      (∀ a, ¬(abi.alloc "out").Contains a → result a=Graph.bits (physical a)) := by
  rw [Execution.Native.execute_eq]
  exact ⟨Machine.closed c abi offset size window domain,persistent_selected dispatch,NativePhysical.returned c abi physical offset size hostScale window complement⟩

/-- Numerical discharge is deliberately separate. `scaleLaw` concerns only
the wrapper's one sqrt/divide scalar; `localLaws` concerns one scalar instruction
or one fixed m16n8k8 MMA output coordinate. Neither assumes an aggregate value. -/
theorem local_numerical_return {α : Type} (meaning : α → ℝ) (actualISA : ISA α)
    (decode : BitVec 32 → ℝ) (zeroWord : decode 0=0)
    (localLaws : LocalTransport meaning actualISA (NativeIdeal.isa decode))
    (c : Shape) (dispatch : Dispatch c) (abi : ABI c) (physical : Nat → BitVec 32)
    (xs : LogicalInputs ℝ c) (normalized : Normalized c abi (fun a => decode (physical a)) xs)
    (offset size : Int) (hostScale : ℝ) (scaleLaw : hostScale=LogicalFormula.hostScale c.D)
    (window complement : Bool) (domain : Representation.Domain c abi offset size window) :
    Machine.Closed c abi offset size window ∧ persistentBlock dispatch=c.B ∧
    ∃ result, Execution.Native.execute c abi physical offset size hostScale window complement=some result ∧
      ∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
        meaning ((result ((abi.alloc "out").base+c.outputBase z.val h.val+j.val)).eval actualISA)=
          RealPhysical.logicalTarget c xs offset size (LogicalFormula.hostScale c.D) window complement z h j := by
  rw [Execution.Native.execute_eq]
  obtain ⟨result,he,hv⟩ := NativePhysical.returned_transport meaning actualISA (NativeIdeal.isa decode) localLaws
    c abi physical offset size hostScale window complement
  refine ⟨Machine.closed c abi offset size window domain,persistent_selected dispatch,result,he,?_⟩
  intro z h j
  rw [hv,NativeReturnedMath.target_logical decode zeroWord c abi physical xs normalized,scaleLaw]

#axiomsClean native_return
#axiomsClean local_numerical_return
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScore.Whole002.FPEquiv
