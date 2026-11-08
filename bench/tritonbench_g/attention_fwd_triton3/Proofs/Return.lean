import bench.tritonbench_g.attention_fwd_triton3.Proofs.Flat
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealGridResult

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Return
open VeriTile.Triton Geometry Host Ownership
open VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
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

/-- END leaves every L cell unchanged in the actual repaired-source schedule. -/
theorem L_preserved (c : Call) (hfinish : c.finish≠0)
    (s : FP.Structural.State Real.R) (off : Nat) :
    (Reference.gridResult c Real.ops s).mem "L" off=s.mem "L" off := by
  apply ArrayLocality.schedule_frame
  intro idx _ lane live equal
  cases lane <;> simp [address,active,hfinish] at live equal

structure View (c : Call) where
  rows : Nat
  output : Nat → Nat → Nat → Nat → ℝ
  center : Nat → ℝ
  denominator : Nat → ℝ
  sameM : Bool
  sameL : Bool
  guaranteesSameO : Bool

def returned (c : Call) (t : FP.Structural.State Real.R) : View c where
  rows:=if c.finish=0 then c.rounded else c.NQ
  output:=fun z h r d=>((t.mem "O" (c.o.at z h r d)).read .real).unbotD 0
  center:=fun i=>((t.mem "M" i).read .real).unbotD 0
  denominator:=fun i=>((t.mem "L" i).read .real).unbotD 0
  sameM:=Bool.true
  sameL:=Bool.true
  guaranteesSameO:=decide (c.finish=0)

theorem identity (c : Call) (t : FP.Structural.State Real.R) :
    (returned c t).sameM=Bool.true ∧ (returned c t).sameL=Bool.true ∧
    (c.finish=0 → (returned c t).guaranteesSameO=Bool.true) := by simp [returned]

/-- Public O coordinates are covered by an explicitly constructed program,
including the padded last tile. The value comes from the new full-grid theorem. -/
theorem output_value (c : Call) (hc : Valid c) (v : RealInterface.Inputs c)
    (t : FP.Structural.State Real.R)
    (hv : ∀ idx lane,active c lane →
      (t.mem (address c idx lane).1 (address c idx lane).2).read .real=
        RealInterface.modelLaneValue c v idx lane)
    (z h r d : Nat) (hz : z<c.Z) (hh : h<c.H) (hr : r<c.NQ) (hd : d<c.D) :
    (returned c t).output z h r d=
      (RealInterface.modelLaneValue c v (owner c hc z h r hz hh hr)
        (.output ⟨r%c.BM,Nat.mod_lt r hc.blockM⟩ ⟨d,hd⟩)).unbotD 0 := by
  let idx:=owner c hc z h r hz hh hr
  let ri : Fin c.BM:=⟨r%c.BM,Nat.mod_lt r hc.blockM⟩
  have value:=congrArg (WithBot.unbotD 0) (hv idx (.output ri ⟨d,hd⟩) trivial)
  have hb : (z*c.H+h)/c.H=z := by
    simp [Nat.mul_comm z c.H,Nat.mul_add_div hc.heads,Nat.div_eq_of_lt hh]
  have hm : (z*c.H+h)%c.H=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have hr' : r/c.BM*c.BM+r%c.BM=r := (row_coverage c.NQ c.BM r hc.blockM hr).2.2
  simpa only [address,Call.outAddress,Call.row,idx,owner_hz,owner_tile,ri,hb,hm,hr',returned] using value

/-- Both launches execute the actual repaired quotation. The second receives
exactly the first final memory, with no reinterpretation of an END carry. -/
theorem two_calls (c d : Call) (hc : Valid c) (hd : Valid d)
    (s : FP.Structural.State Real.R) :
    ∃ middle final,
      Reference.Composition.schedule Real.ops.algebra
        (Reference.launched c) c.grid (Reference.Composition.allPrograms c.grid) s=some middle ∧
      middle.mem=(Reference.gridResult c Real.ops s).mem ∧
      Reference.Composition.schedule Real.ops.algebra
        (Reference.launched d) d.grid (Reference.Composition.allPrograms d.grid) middle=some final ∧
      final.mem=(Reference.gridResult d Real.ops middle).mem := by
  obtain ⟨middle,hm,values⟩:=Reference.complete_grid c Real.ops hc.blockN s
  obtain ⟨final,hf,values'⟩:=Reference.complete_grid d Real.ops hd.blockN middle
  exact ⟨middle,final,hm,values,hf,values'⟩

#axiomsClean output_covered
#axiomsClean output_value
#axiomsClean L_preserved
#axiomsClean two_calls
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Return
