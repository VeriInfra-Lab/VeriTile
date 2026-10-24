import bench.repairs.tritonbench_g.attention_score.Proofs.NativeLoop
import bench.repairs.tritonbench_g.attention_score.Proofs.Physical

namespace VeriTile.Bench.Repairs.AttentionScore.NativePhysical
open VeriTile Triton FP.Structural NativeInstructions NativeStaging Geometry Physical
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

def input (physical : Nat → BitVec 32) : Nat → Expr := fun a => .ready (.bits (physical a))
def execute (c : Shape) (abi : ABI c) (physical : Nat → BitVec 32)
    (offset size : Int) (scale : ℝ) (window complement : Bool) : Option (Nat → Graph) :=
  (Physical.execute algebra dotModel c abi (input physical) offset size scale window complement).map
    (fun result a => lower (result a))

def target (c : Shape) (abi : ABI c) (physical : Nat → BitVec 32)
    (offset size : Int) (scale : ℝ) (window complement : Bool)
    (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) : Graph :=
  NativeLoop.returnedGraph c (Physical.initial abi (input physical)) offset size scale window complement z h j

/-- Full actual source execution at the source-specific native lowering,
including allocation, all grid jobs, cast, finite readback, frame and empty return. -/
theorem returned (c : Shape) (abi : ABI c) (physical : Nat → BitVec 32)
    (offset size : Int) (scale : ℝ) (window complement : Bool) :
    ∃ result, execute c abi physical offset size scale window complement=some result ∧
      (∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
        result ((abi.alloc "out").base+c.outputBase z.val h.val+j.val)=target c abi physical offset size scale window complement z h j) ∧
      (∀ (r : RegionName), tracked r → r≠"out" → ∀ i, i<(abi.alloc r).length →
        result ((abi.alloc r).base+i)=Graph.bits (physical ((abi.alloc r).base+i))) ∧
      (∀ a, ¬(abi.alloc "out").Contains a → result a=Graph.bits (physical a)) := by
  obtain ⟨result,he,hv,hp,hf⟩ := Physical.returned algebra dotModel c abi (input physical) offset size scale window complement
  refine ⟨fun a => lower (result a),?_,?_,?_,?_⟩
  · simp [execute,he]
  · intro z h j
    dsimp only
    rw [hv]
    exact NativeLoop.returned_lower _ _ _ _ _ _ _ _ _ _
  · intro r hr hne i hi
    dsimp only
    rw [hp r hr hne i hi]
    rfl
  · intro a ha
    dsimp only
    rw [hf a ha]
    rfl

/-- The only numerical premise is the fixed scalar/fixed-MMA transport record.
It contains no whole-dot, reduction, program-output or success assumption. -/
theorem returned_transport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β)
    (localLaws : LocalTransport f A B) (c : Shape) (abi : ABI c) (physical : Nat → BitVec 32)
    (offset size : Int) (scale : ℝ) (window complement : Bool) :
    ∃ result, execute c abi physical offset size scale window complement=some result ∧
      ∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
        f ((result ((abi.alloc "out").base+c.outputBase z.val h.val+j.val)).eval A)=
          (target c abi physical offset size scale window complement z h j).eval B := by
  obtain ⟨result,he,hv,_,_⟩ := returned c abi physical offset size scale window complement
  refine ⟨result,he,?_⟩
  intro z h j
  rw [hv]
  exact transport f A B localLaws _

#specNonCircular target avoiding [execute,Physical.execute,FP.Structural.exec,FPReturned.hostRun]
#axiomsClean returned
#axiomsClean returned_transport
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScore.NativePhysical
