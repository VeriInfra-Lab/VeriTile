import bench.tritonbench_g.int_scaled_matmul.IO

namespace VeriTile.Bench.Repairs.IntScaledMatmul
open VeriTile.Triton FP.Structural Revision001
noncomputable section
local instance : Inhabited KernelModel.Word := ⟨FP.Value.zero .fp32⟩

namespace IntIO
theorem schedule_equal (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters)
    (pids : List Nat) (s : State KernelModel.Word) :
    runSchedule (pids.map (invocation rest p)) s=
      GridExecution.sequence (BlockWord.one rest p) pids s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid pids ih =>
    change ((BlockWord.one rest p pid s).bind
      (runSchedule (pids.map (invocation rest p))))=_
    simp only [GridExecution.sequence]
    cases run : BlockWord.one rest p pid s with
    | none => rfl
    | some next => exact ih next

theorem formula_bound (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters)
    (vals : (io rest p).Inputs) (initial : State KernelModel.Word)
    (h : (io rest p).InputBinding vals initial) (j : TileIndex [p.rows,p.columns]) :
    BlockWord.logicalDot p initial j.1.val j.2.1.val=formula p vals (0 : Fin 1) j := by
  change (BitVec.ofInt 32 (LogicalSum.sum (BlockWord.logicalProduct p initial j.1.val j.2.1.val) p.inner)).toInt=
    Formula.dot (vals (0 : Fin 2)) (vals (1 : Fin 2)) j
  unfold Formula.dot
  congr 2
  apply LogicalSum.sum_congr
  intro k hk
  have live : 0<p.rows ∧ 0<p.columns :=
    ⟨lt_of_le_of_lt (Nat.zero_le _) j.1.isLt,lt_of_le_of_lt (Nat.zero_le _) j.2.1.isLt⟩
  have a := h (0 : Fin 2) (j.1,⟨k,hk⟩,PUnit.unit) live
  have b := h (1 : Fin 2) (⟨k,hk⟩,j.2.1,PUnit.unit) live
  change (initial.mem p.input (j.1.val*p.inputRowStride+k*p.inputInnerStride)).read .int=
    vals (0 : Fin 2) (j.1,⟨k,hk⟩,PUnit.unit) at a
  change (initial.mem p.weight (k*p.weightInnerStride+j.2.1.val*p.weightColumnStride)).read .int=
    vals (1 : Fin 2) (⟨k,hk⟩,j.2.1,PUnit.unit) at b
  simp only [BlockWord.logicalProduct,BlockWord.wrap,dif_pos hk,a,b]

/-- The existing complete-grid theorem supplies every typed output and the
entire exterior cell frame, for all memories satisfying the finite IO binding. -/
theorem result (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters)
    (geometry : Geometry.Valid (BlockWord.geometry p)) (positive : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (vals : (io rest p).Inputs) (initial : State KernelModel.Word)
    (hin : (io rest p).InputBinding vals initial) :
    ∃ final,runSchedule ((io rest p).schedule vals) initial=some final ∧
      (io rest p).OutputBinding vals (formula p vals) final ∧
      (io rest p).Frame vals initial final := by
  obtain ⟨final,executed,values,frame⟩ := BlockWord.grid_correct rest p geometry positive ha hb initial
  refine ⟨final,(schedule_equal rest p _ initial).trans executed,?_,?_⟩
  · intro o j _
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    have output := values (j.1,j.2.1) trivial
    change mapCell id (final.mem p.output
      (j.1.val*p.outputRowStride+j.2.1.val*p.outputColumnStride))=
      .mk .int (formula p vals (0 : Fin 1) j)
    simp only [Geometry.address,BlockWord.geometry,ScaledWord.geometry] at output
    rw [output]
    change Cell.mk .int (BlockWord.logicalDot p initial j.1.val j.2.1.val)=_
    rw [formula_bound rest p vals initial hin j]
  · intro r offset outside
    apply frame r offset
    by_cases same : r=p.output
    · right
      intro j _ equal
      have no := outside.1 (0 : Fin 1) (j.1,j.2,PUnit.unit) trivial
      exact no.elim (fun h => h same) (fun h => h equal)
    · exact Or.inl same
end IntIO

namespace ScaledIO
theorem schedule_equal (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) (pids : List Nat) (s : State KernelModel.Word) :
    runSchedule (pids.map (invocation rest undefined p)) s=
      GridExecution.sequence (ScaledWord.one rest Native.sourceConfig (KernelModel.fallback undefined) p) pids s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid pids ih =>
    change ((ScaledWord.one rest Native.sourceConfig (KernelModel.fallback undefined) p pid s).bind
      (runSchedule (pids.map (invocation rest undefined p))))=_
    simp only [GridExecution.sequence]
    cases run : ScaledWord.one rest Native.sourceConfig (KernelModel.fallback undefined) p pid s with
    | none => rfl
    | some next => exact ih next

theorem dot_bound (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) (vals : (io rest undefined p).Inputs) (initial : State KernelModel.Word)
    (h : (io rest undefined p).InputBinding vals initial) (j : TileIndex [p.rows,p.columns]) :
    ScaledWord.logicalDot p initial j.1.val j.2.1.val=
      Formula.dot (vals (0 : Fin 3)) (vals (1 : Fin 3)) j := by
  unfold ScaledWord.logicalDot ScaledWord.wrap Formula.dot
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
  simp only [ScaledWord.logicalProduct,ScaledWord.wrap,dif_pos hk,a,b]

theorem formula_bound (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) (vals : (io rest undefined p).Inputs) (initial : State KernelModel.Word)
    (h : (io rest undefined p).InputBinding vals initial) (j : TileIndex [p.rows,p.columns]) :
    TotalScaled.logicalResult Native.sourceConfig (KernelModel.fallback undefined) p initial j.1.val j.2.1.val=
      formula undefined p vals (0 : Fin 1) j := by
  have live : 0<p.rows ∧ 0<p.columns :=
    ⟨lt_of_le_of_lt (Nat.zero_le _) j.1.isLt,lt_of_le_of_lt (Nat.zero_le _) j.2.1.isLt⟩
  have scale := h (2 : Fin 3) (j.1,PUnit.unit) live
  change (initial.mem p.scale (j.1.val*p.scaleRowStride)).read .real=
    vals (2 : Fin 3) (j.1,PUnit.unit) at scale
  change FP.IntegerConversion.totalSigned (KernelModel.fallback undefined 32) 32
    (FP.mul Native.sourceConfig
      (FP.round .fp32 Native.sourceConfig (ScaledWord.logicalDot p initial j.1.val j.2.1.val : Rat))
      ((initial.mem p.scale (j.1.val*p.scaleRowStride)).read .real))=_
  rw [dot_bound rest undefined p vals initial h j,scale]
  rfl

theorem result (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) (geometry : Geometry.Valid (ScaledWord.geometry p))
    (even : p.even=Bool.true → p.inner=p.steps*p.blockInner) (cover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output)
    (vals : (io rest undefined p).Inputs) (initial : State KernelModel.Word)
    (hin : (io rest undefined p).InputBinding vals initial) :
    ∃ final,runSchedule ((io rest undefined p).schedule vals) initial=some final ∧
      (io rest undefined p).OutputBinding vals (formula undefined p vals) final ∧
      (io rest undefined p).Frame vals initial final := by
  obtain ⟨final,executed,values,frame⟩ := TotalScaled.grid_correct rest Native.sourceConfig
    (KernelModel.fallback undefined) p geometry even cover ha hb hs initial
  refine ⟨final,(schedule_equal rest undefined p _ initial).trans executed,?_,?_⟩
  · intro o j _
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    have output := values (j.1,j.2.1) trivial
    change mapCell id (final.mem p.output
      (j.1.val*p.outputRowStride+j.2.1.val*p.outputColumnStride))=
      .mk .int (formula undefined p vals (0 : Fin 1) j)
    simp only [Geometry.address,BlockWord.geometry,ScaledWord.geometry] at output
    rw [output]
    change Cell.mk .int (TotalScaled.logicalResult Native.sourceConfig
      (KernelModel.fallback undefined) p initial j.1.val j.2.1.val)=_
    rw [formula_bound rest undefined p vals initial hin j]
  · intro r offset outside
    apply frame r offset
    by_cases same : r=p.output
    · right
      intro j _ equal
      have no := outside.1 (0 : Fin 1) (j.1,j.2,PUnit.unit) trivial
      exact no.elim (fun h => h same) (fun h => h equal)
    · exact Or.inl same
end ScaledIO

#axiomsClean IntIO.result
#axiomsClean ScaledIO.result
end
end VeriTile.Bench.Repairs.IntScaledMatmul
