import bench.tritonbench_g.attention_score.Proofs.Reused.Physical
import bench.tritonbench_g.attention_score.Proofs.Reused.RealReturned

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealPhysical
open VeriTile Triton Geometry FPStorage Physical
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

def initial (abi : ABI c) (physical : Nat → ℝ) : BlockState where
  mem := fun r a => .real (physical ((abi.alloc r).base+a))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun dim => if dim=0 then c.programs else if dim=1 then c.Z*c.H else 1

def execute (c : Shape) (abi : ABI c) (physical : Nat → ℝ)
    (offset size : Int) (scale : ℝ) (window complement : Bool) : Option (Nat → ℝ) :=
  (RealReturned.hostRun c (initial abi physical) offset size scale window complement).map
    (fun t => flush (abi.alloc "out") (t.readMem "out") physical)

/-- The independent full-tensor mathematical target has exactly N terms,
uses the supplied M, and does not mention blocks, an interpreter or an output. -/
def logicalTarget (c : Shape) (xs : LogicalInputs ℝ c) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) : ℝ :=
  ∑ i : Fin c.N, if RealFormula.windowAllows window complement offset size i.val j.val then
    pow2 ((∑ d : Fin c.D, xs.q z h i d * xs.k z ⟨h.val/(c.H/c.HK),grouped_head c h⟩ j d) *
      (scale*1.4426950408889634) - xs.m z h ⟨i.val,lt_of_lt_of_le i.isLt c.normalizerCapacity⟩)
  else 0

theorem target_logical (c : Shape) (abi : ABI c) (physical : Nat → ℝ) (xs : LogicalInputs ℝ c)
    (normalized : Normalized c abi physical xs) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) :
    RealReturned.target c (initial abi physical) offset size scale window complement z h j=
      logicalTarget c xs offset size scale window complement z h j := by
  unfold RealReturned.target LogicalFormula.column logicalTarget
  apply Finset.sum_congr rfl
  intro i _
  have hq (d : Fin c.D) :
      (DenseSource.inputs c (initial abi physical) z.val h.val).query i.val d=xs.q z h i d := by
    simpa [DenseSource.inputs,RealLoads.inputs,initial,BlockState.readMem,Nat.mul_one,Nat.add_assoc] using normalized.query z h i d
  have hk (d : Fin c.D) :
      (DenseSource.inputs c (initial abi physical) z.val h.val).key j.val d=
        xs.k z ⟨h.val/(c.H/c.HK),grouped_head c h⟩ j d := by
    simpa [DenseSource.inputs,RealLoads.inputs,initial,BlockState.readMem,Shape.keyBase,
      Nat.mul_one,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
      normalized.key z ⟨h.val/(c.H/c.HK),grouped_head c h⟩ j d
  have hm : (DenseSource.inputs c (initial abi physical) z.val h.val).normalizer i.val=
      xs.m z h ⟨i.val,lt_of_lt_of_le i.isLt c.normalizerCapacity⟩ := by
    simpa [DenseSource.inputs,RealLoads.inputs,initial,BlockState.readMem,Nat.add_assoc] using
      normalized.normalizer z h ⟨i.val,lt_of_lt_of_le i.isLt c.normalizerCapacity⟩
  simp only [LogicalFormula.weight,hq,hk,hm]

theorem returned (c : Shape) (keyRange : c.N≠0 → c.outputSize≠0 → c.programs*c.B<2147483648) (abi : ABI c) (physical : Nat → ℝ) (xs : LogicalInputs ℝ c)
    (normalized : Normalized c abi physical xs) (offset size : Int) (scale : ℝ)
    (window complement : Bool) :
    ∃ result, execute c abi physical offset size scale window complement=some result ∧
      (∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
        result ((abi.alloc "out").base+c.outputBase z.val h.val+j.val)=
          logicalTarget c xs offset size scale window complement z h j) ∧
      (∀ (r : RegionName), tracked r → r≠"out" → ∀ i, i<(abi.alloc r).length →
        result ((abi.alloc r).base+i)=physical ((abi.alloc r).base+i)) ∧
      (∀ a, ¬(abi.alloc "out").Contains a → result a=physical a) := by
  obtain ⟨t,ht,_,hv,_⟩ := RealReturned.host_returned c keyRange (initial abi physical) offset size scale window complement
  refine ⟨flush (abi.alloc "out") (t.readMem "out") physical,?_,?_,?_,?_⟩
  · simp [execute,ht]
  · intro z h j
    have hlen : (abi.alloc "out").length=c.outputSize := by
      simpa [Physical.size] using abi.lengths "out" (by simp [tracked])
    have hi : c.outputBase z.val h.val+j.val<(abi.alloc "out").length := by
      rw [hlen]; exact output_bound c z h j
    rw [Nat.add_assoc,flush_readback _ _ _ _ hi,hv]
    exact target_logical c abi physical xs normalized offset size scale window complement z h j
  · intro r hr hne i hi
    exact flush_preserves_input _ _ (abi.fresh r hr hne) _ _ i hi
  · intro a ha
    exact flush_frame _ _ _ a ha

#specNonCircular logicalTarget avoiding [execute,RealReturned.hostRun,WholeGrid.run,VeriTile.Triton.exec]
#axiomsClean returned
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealPhysical
