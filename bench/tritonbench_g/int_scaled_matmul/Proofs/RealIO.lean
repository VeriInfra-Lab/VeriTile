import bench.tritonbench_g.int_scaled_matmul.IO
import bench.tritonbench_g.int_scaled_matmul.Proofs.RealSafety
import bench.tritonbench_g.int_scaled_matmul.Proofs.IOComplete

namespace VeriTile.Bench.Repairs.IntScaledMatmul.RealIO
open VeriTile.Triton FP.Structural Revision001
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 28000
theorem schedule_equal (rest : Algebra ℝ)
    (p : ScaledWord.Parameters) (pids : List Nat) (s : State ℝ) :
    runSchedule (pids.map (invocation rest p)) s=
      GridExecution.sequence (RealScaled.one rest p) pids s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid pids ih =>
    change ((RealScaled.one rest p pid s).bind
      (runSchedule (pids.map (invocation rest p))))=_
    simp only [GridExecution.sequence]
    cases run : RealScaled.one rest p pid s with
    | none => rfl
    | some next => exact ih next

theorem dot_bound (rest : Algebra ℝ)
    (p : ScaledWord.Parameters) (vals : (io rest p).Inputs) (initial : State ℝ)
    (h : (io rest p).InputBinding vals initial) (j : TileIndex [p.rows,p.columns]) :
    RealScaled.logicalDot p initial j.1.val j.2.1.val=
      Formula.dot (vals (0 : Fin 3)) (vals (1 : Fin 3)) j := by
  unfold RealScaled.logicalDot RealScaled.wrap Formula.dot
  congr 2
  apply LogicalSum.sum_congr
  intro k hk
  have live : 0<p.rows ∧ 0<p.columns :=
    ⟨lt_of_le_of_lt (Nat.zero_le _) j.1.isLt,lt_of_le_of_lt (Nat.zero_le _) j.2.1.isLt⟩
  have a := h (0 : Fin 3) (j.1,⟨k,hk⟩,PUnit.unit) live
  have b := h (1 : Fin 3) (⟨k,hk⟩,j.2.1,PUnit.unit) live
  change (initial.mem p.input (j.1.val*p.inputRowStride+k*p.inputInnerStride)).read .int=
    vals (0 : Fin 3) (j.1,⟨k,hk⟩,PUnit.unit) at a
  change (initial.mem p.weight (k*p.weightInnerStride+j.2.1.val*p.weightColumnStride)).read .int=
    vals (1 : Fin 3) (⟨k,hk⟩,j.2.1,PUnit.unit) at b
  simp only [RealScaled.logicalProduct,RealScaled.wrap,dif_pos hk,a,b]

theorem formula_bound (rest : Algebra ℝ) (p : ScaledWord.Parameters)
    (vals : (io rest p).Inputs) (initial : State ℝ)
    (h : (io rest p).InputBinding vals initial) (j : TileIndex [p.rows,p.columns]) :
    RealScaled.logicalResult p initial j.1.val j.2.1.val=formula p vals (0 : Fin 1) j := by
  have live : 0<p.rows ∧ 0<p.columns :=
    ⟨lt_of_le_of_lt (Nat.zero_le _) j.1.isLt,lt_of_le_of_lt (Nat.zero_le _) j.2.1.isLt⟩
  have scale := h (2 : Fin 3) (j.1,PUnit.unit) live
  change (initial.mem p.scale (j.1.val*p.scaleRowStride)).read .real=
    vals (2 : Fin 3) (j.1,PUnit.unit) at scale
  change RealScaled.wrap 32 (RealScaled.truncate
    ((RealScaled.logicalDot p initial j.1.val j.2.1.val : ℝ)*
      (initial.mem p.scale (j.1.val*p.scaleRowStride)).read .real))=_
  rw [dot_bound rest p vals initial h j,scale]
  rfl
theorem result (rest : Algebra ℝ)
    (p : ScaledWord.Parameters) (geometry : Geometry.Valid (ScaledWord.geometry p))
    (even : p.even=Bool.true → p.inner=p.steps*p.blockInner) (cover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output)
    (vals : (io rest p).Inputs) (initial : State ℝ)
    (hin : (io rest p).InputBinding vals initial) :
    ∃ final,runSchedule ((io rest p).schedule vals) initial=some final ∧
      (io rest p).OutputBinding vals (formula p vals) final ∧
      (io rest p).Frame vals initial final := by
  obtain ⟨final,executed,values,frame⟩ := RealScaled.grid_correct rest p geometry even cover ha hb hs initial
  refine ⟨final,(schedule_equal rest p _ initial).trans executed,?_,?_⟩
  · intro o j _
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    have output := values (j.1,j.2.1) trivial
    change mapCell id (final.mem p.output
      (j.1.val*p.outputRowStride+j.2.1.val*p.outputColumnStride))=
      .mk .int (formula p vals (0 : Fin 1) j)
    simp only [Geometry.address,RealScaled.geometry] at output
    rw [output]
    change Cell.mk .int (RealScaled.logicalResult p initial j.1.val j.2.1.val)=_
    rw [formula_bound rest p vals initial hin j]
  · intro r offset outside
    apply frame r offset
    by_cases same : r=p.output
    · right
      intro j _ equal
      have no := outside.1 (0 : Fin 1) (j.1,j.2,PUnit.unit) trivial
      exact no.elim (fun h => h same) (fun h => h equal)
    · exact Or.inl same
theorem single_safe (rest : Algebra ℝ)
    (p : ScaledWord.Parameters)
    (even : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (nonempty : 0<p.rows ∧ 0<p.columns) (bounds : RegionName → Nat)
    (ha : ∀ r k,r<p.rows → k<p.inner →
      r*p.inputRowStride+k*p.inputInnerStride<bounds p.input)
    (hb : ∀ k c,k<p.inner → c<p.columns →
      k*p.weightInnerStride+c*p.weightColumnStride<bounds p.weight)
    (hs : ∀ r,r<p.rows → r*p.scaleRowStride<bounds p.scale)
    (ho : ∀ r c,r<p.rows → c<p.columns →
      r*p.outputRowStride+c*p.outputColumnStride<bounds p.output)
    (initial : State ℝ) :
    TraceSafe (RealScaled.backend rest)
      bounds (ScaledWord.source p) initial := by
  apply RealScaledSafety.complete
  · intro k hk i live
    have inner := ScaledIO.active_inner p even k hk i.2.1 live
    have bound := ha (RealScaled.inputRow p initial i.1) (k*p.blockInner+i.2.1.val)
      (Nat.mod_lt _ nonempty.1) inner
    simpa only [RealScaled.aOffset,RealScaled.inputInner,Nat.add_mul,Nat.mul_assoc,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using bound
  · intro k hk i live
    have inner := ScaledIO.active_inner p even k hk i.1 live
    have bound := hb (k*p.blockInner+i.1.val) (RealScaled.inputColumn p initial i.2.1)
      inner (Nat.mod_lt _ nonempty.2)
    simpa only [RealScaled.bOffset,RealScaled.inputInner,Nat.add_mul,Nat.mul_assoc,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using bound
  · intro i hi
    exact hs _ hi.1
  · intro i hi
    simpa only [RealScaled.outputOffset,Nat.add_comm] using ho _ _ hi.1 hi.2

theorem schedule_safe (rest : Algebra ℝ)
    (p : ScaledWord.Parameters) (geometry : Geometry.Valid (ScaledWord.geometry p))
    (even : p.even=Bool.true → p.inner=p.steps*p.blockInner) (bounds : RegionName → Nat)
    (ha : ∀ r k,r<p.rows → k<p.inner →
      r*p.inputRowStride+k*p.inputInnerStride<bounds p.input)
    (hb : ∀ k c,k<p.inner → c<p.columns →
      k*p.weightInnerStride+c*p.weightColumnStride<bounds p.weight)
    (hs : ∀ r,r<p.rows → r*p.scaleRowStride<bounds p.scale)
    (ho : ∀ r c,r<p.rows → c<p.columns →
      r*p.outputRowStride+c*p.outputColumnStride<bounds p.output)
    (pids : List Nat) (reached : ∀ pid∈pids,pid<Geometry.count (ScaledWord.geometry p))
    (initial : State ℝ) :
    ScheduleSafe bounds (pids.map (invocation rest p)) initial := by
  induction pids generalizing initial with
  | nil => trivial
  | cons pid pids ih =>
    refine ⟨?_,fun next _ => ih (fun q hq => reached q (by simp [hq])) next⟩
    exact single_safe rest p even
      (ScaledIO.reached_nonempty p geometry pid (reached pid (by simp))) bounds ha hb hs ho
      (GridExecution.atProgram initial pid)

theorem complete (rest : Algebra ℝ)
    (p : ScaledWord.Parameters) (geometry : Geometry.Valid (ScaledWord.geometry p))
    (even : p.even=Bool.true → p.inner=p.steps*p.blockInner) (cover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) :
    (io rest p).Implements (formula p) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hbound initial _
    by_cases live : 0<p.rows ∧ 0<p.columns
    · apply schedule_safe rest p geometry even bounds
      · intro r k hr hk
        exact hbound.2.1 (0 : Fin 3) (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) live
      · intro k c hk hc
        exact hbound.2.1 (1 : Fin 3) (⟨k,hk⟩,⟨c,hc⟩,PUnit.unit) live
      · intro r hr
        exact hbound.2.1 (2 : Fin 3) (⟨r,hr⟩,PUnit.unit) live
      · intro r c hr hc
        exact hbound.2.2.1 (0 : Fin 1) (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) trivial
      · intro pid hp
        exact List.mem_range.mp hp
    · have zero := empty_count (ScaledWord.geometry p) geometry live
      change ScheduleSafe bounds ((List.range (Geometry.count (ScaledWord.geometry p))).map (invocation rest p)) initial
      rw [zero]
      trivial
  · exact result rest p geometry even cover ha hb hs
#axiomsClean complete
end
end VeriTile.Bench.Repairs.IntScaledMatmul.RealIO
