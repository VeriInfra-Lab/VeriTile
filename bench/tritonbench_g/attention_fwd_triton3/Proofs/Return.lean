import bench.tritonbench_g.attention_fwd_triton3.Proofs.Flat

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Return
open VeriTile.Triton Geometry Host Ownership
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section

def owner (c : Call) (hc : Valid c) (z h r : Nat)
    (hz : z<c.Z) (hh : h<c.H) (hr : r<c.NQ) : GridIndex c.grid := by
  intro ax
  by_cases ha : ax.val=0
  · refine ⟨r/c.BM,?_⟩
    simpa [Call.grid,Grid.dim,ha] using (row_coverage c.NQ c.BM r hc.blockM hr).1
  · have ha1 : ax.val=1 := by have hi : ax.val<2:=ax.isLt; omega
    refine ⟨z*c.H+h,?_⟩
    have hb:=Nat.mul_le_mul_right c.H (show z+1≤c.Z by omega)
    have ht : z*c.H+h<c.Z*c.H := by nlinarith
    simpa [Call.grid,Grid.dim,ha1] using ht

theorem owner_tile (c : Call) (hc : Valid c) (z h r : Nat)
    (hz : z<c.Z) (hh : h<c.H) (hr : r<c.NQ) : c.tile (owner c hc z h r hz hh hr)=r/c.BM := rfl

theorem owner_hz (c : Call) (hc : Valid c) (z h r : Nat)
    (hz : z<c.Z) (hh : h<c.H) (hr : r<c.NQ) : c.hz (owner c hc z h r hz hh hr)=z*c.H+h := rfl

theorem output_covered (c : Call) (hc : Valid c) (z h r d : Nat)
    (hz : z<c.Z) (hh : h<c.H) (hr : r<c.NQ) (hd : d<c.D) :
    ∃ idx : GridIndex c.grid,∃ i : Lane c.BM c.D,
      active c i ∧ address c idx i=(("O" : RegionName),c.o.at z h r d) := by
  let idx:=owner c hc z h r hz hh hr
  let ri : Fin c.BM:=⟨r%c.BM,Nat.mod_lt r hc.blockM⟩
  refine ⟨idx,.output ri ⟨d,hd⟩,trivial,?_⟩
  have hb : (z*c.H+h)/c.H=z := by
    simp [Nat.mul_comm z c.H,Nat.mul_add_div hc.heads,Nat.div_eq_of_lt hh]
  have hm : (z*c.H+h)%c.H=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have hr' : r/c.BM*c.BM+r%c.BM=r := (row_coverage c.NQ c.BM r hc.blockM hr).2.2
  simp only [address,Call.outAddress,Call.row,idx,owner_hz,owner_tile,ri,hb,hm,hr']

/-- Raw L storage is unchanged by END, including padding and unrelated cells. -/
theorem L_preserved (c : Call) (hfinish : c.finish≠0) (s t : BlockState)
    (hf : BlockState.WriteWithin (Whole.writes c) s t) (off : Nat) : t.mem "L" off=s.mem "L" off := by
  symm
  apply hf
  rintro ⟨idx,i,hi,he⟩
  cases i <;> simp [address,active,hfinish] at hi he

/-- Host public slice/contiguous/to(FP32) value view. Object identity for M/L
is unconditional; END=false also returns the supplied O object. For END=true,
Torch may reuse or allocate O, so no fresh-storage assertion is made. -/
structure View (c : Call) where
  rows : Nat
  output : Nat → Nat → Nat → Nat → ℝ
  center : Nat → ℝ
  denominator : Nat → ℝ
  sameM : Bool
  sameL : Bool
  guaranteesSameO : Bool

def returned (c : Call) (t : BlockState) : View c where
  rows := if c.finish=0 then c.rounded else c.NQ
  output := fun z h r d=>t.readMem "O" (c.o.at z h r d)
  center := t.readMem "M"
  denominator := t.readMem "L"
  sameM := Bool.true
  sameL := Bool.true
  guaranteesSameO := decide (c.finish=0)

theorem identity (c : Call) (t : BlockState) :
    (returned c t).sameM=Bool.true ∧ (returned c t).sameL=Bool.true ∧
    (c.finish=0 → (returned c t).guaranteesSameO=Bool.true) := by simp [returned]

/-- Complete caller-visible O formula, read from the actual completed source
launch. Its owner is constructed from row division, never supplied as a premise. -/
theorem output_value (c : Call) (hc : Valid c) (s t : BlockState)
    (hv : ∀ idx i,active c i → t.readMem (address c idx i).1 (address c idx i).2=Whole.result c s idx i)
    (z h r d : Nat) (hz : z<c.Z) (hh : h<c.H) (hr : r<c.NQ) (hd : d<c.D) :
    (returned c t).output z h r d =
      (Whole.expected c s (owner c hc z h r hz hh hr)).output
        (⟨r%c.BM,Nat.mod_lt r hc.blockM⟩,⟨d,hd⟩,PUnit.unit) := by
  let idx:=owner c hc z h r hz hh hr
  let ri : Fin c.BM:=⟨r%c.BM,Nat.mod_lt r hc.blockM⟩
  have hval:=hv idx (.output ri ⟨d,hd⟩) trivial
  have hb : (z*c.H+h)/c.H=z := by
    simp [Nat.mul_comm z c.H,Nat.mul_add_div hc.heads,Nat.div_eq_of_lt hh]
  have hm : (z*c.H+h)%c.H=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have hr' : r/c.BM*c.BM+r%c.BM=r := (row_coverage c.NQ c.BM r hc.blockM hr).2.2
  simpa only [address,Call.outAddress,Call.row,idx,owner_hz,owner_tile,ri,hb,hm,hr',
    Whole.result,returned] using hval

/-- Two successful complete calls can be chained with the actual first final
memory as the second's six-input state. This includes arbitrary raw carry;
END=true is not relabeled as an unnormalized streaming state. -/
theorem two_calls (c d : Call) (hc : Valid c) (hd : Valid d)
    (ac : Allocated c extent) (ad : Allocated d extent) (s : BlockState) :
    ∃ middle final,
      Nonempty (Kernel.GridLaunchedOrdinary c.source c.grid s middle) ∧
      (∀ idx i,active c i → middle.readMem (address c idx i).1 (address c idx i).2=Whole.result c s idx i) ∧
      Nonempty (Kernel.GridLaunchedOrdinary d.source d.grid middle final) ∧
      (∀ idx i,active d i → final.readMem (address d idx i).1 (address d idx i).2=Whole.result d middle idx i) := by
  obtain ⟨_,middle,hm,hmv,_⟩:=Whole.complete c hc ac s
  obtain ⟨_,final,hf,hfv,_⟩:=Whole.complete d hd ad middle
  exact ⟨middle,final,hm,hmv,hf,hfv⟩

#print axioms output_covered
#print axioms two_calls
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Return
