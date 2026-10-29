import bench.repairs.tritonbench_g.int_scaled_matmul.CandidateIO
import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.CandidateExecution

namespace VeriTile.Bench.Repairs.IntScaledMatmul
open VeriTile.Triton FP.Structural Revision001
noncomputable section
local instance : Inhabited KernelModel.Word := ⟨FP.Value.zero .fp32⟩
namespace CandidateIntIO
abbrev formula := IntIO.formula
abbrev formula_bound := IntIO.formula_bound
theorem schedule_equal (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters)
    (pids : List Nat) (s : State KernelModel.Word) :
    runSchedule (pids.map (invocation rest p)) s=
      GridExecution.sequence (CopyFree.one rest p) pids s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid pids ih =>
    change ((CopyFree.one rest p pid s).bind
      (runSchedule (pids.map (invocation rest p))))=_
    simp only [GridExecution.sequence]
    cases run : CopyFree.one rest p pid s with
    | none => rfl
    | some next => exact ih next

theorem result (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters)
    (geometry : Geometry.Valid (BlockWord.geometry p)) (positive : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (vals : (io rest p).Inputs) (initial : State KernelModel.Word)
    (hin : (io rest p).InputBinding vals initial) :
    ∃ final,runSchedule ((io rest p).schedule vals) initial=some final ∧
      (io rest p).OutputBinding vals (formula p vals) final ∧
      (io rest p).Frame vals initial final := by
  obtain ⟨final,executed,values,frame⟩ := CopyFree.grid_correct rest p geometry positive ha hb initial
  refine ⟨final,(schedule_equal rest p _ initial).trans executed,?_,?_⟩
  · intro o j _
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    have output := values (j.1,j.2.1) trivial
    change mapCell id (final.mem p.output
      (j.1.val*p.outputRowStride+j.2.1.val*p.outputColumnStride))=
      .mk .int (formula p vals (0 : Fin 1) j)
    simp only [Geometry.address,BlockWord.geometry] at output
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
theorem schedule_safe (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters)
    (positive : 0<p.blockInner) (bounds : RegionName → Nat)
    (ha : ∀ r k,r<p.rows → k<p.inner →
      r*p.inputRowStride+k*p.inputInnerStride<bounds p.input)
    (hb : ∀ k c,k<p.inner → c<p.columns →
      k*p.weightInnerStride+c*p.weightColumnStride<bounds p.weight)
    (ho : ∀ r c,r<p.rows → c<p.columns →
      r*p.outputRowStride+c*p.outputColumnStride<bounds p.output)
    (pids : List Nat) (initial : State KernelModel.Word) :
    ScheduleSafe bounds (pids.map (invocation rest p)) initial := by
  induction pids generalizing initial with
  | nil => trivial
  | cons pid pids ih =>
    exact ⟨CopyFree.safe rest p positive bounds ha hb ho (GridExecution.atProgram initial pid),
      fun _ _ => ih _⟩

theorem complete (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters)
    (geometry : Geometry.Valid (BlockWord.geometry p)) (positive : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    (io rest p).Implements (formula p) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hbound initial _
    by_cases live : 0<p.rows ∧ 0<p.columns
    · apply schedule_safe rest p positive bounds
      · intro r k hr hk
        exact hbound.2.1 (0 : Fin 2) (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) live
      · intro k c hk hc
        exact hbound.2.1 (1 : Fin 2) (⟨k,hk⟩,⟨c,hc⟩,PUnit.unit) live
      · intro r c hr hc
        exact hbound.2.2.1 (0 : Fin 1) (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) trivial
    · have zero := empty_count (BlockWord.geometry p) geometry live
      change ScheduleSafe bounds ((List.range (Geometry.count (BlockWord.geometry p))).map (invocation rest p)) initial
      rw [zero]
      trivial
  · exact result rest p geometry positive ha hb
end CandidateIntIO

namespace CandidateScaledIO
abbrev formula := ScaledIO.formula
abbrev formula_bound := ScaledIO.formula_bound
abbrev reached_nonempty := ScaledIO.reached_nonempty
abbrev active_inner := ScaledIO.active_inner
theorem schedule_equal (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) (pids : List Nat) (s : State KernelModel.Word) :
    runSchedule (pids.map (invocation rest undefined p)) s=
      GridExecution.sequence (SharedIndices.one rest Native.sourceConfig (KernelModel.fallback undefined) p) pids s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid pids ih =>
    change ((SharedIndices.one rest Native.sourceConfig (KernelModel.fallback undefined) p pid s).bind
      (runSchedule (pids.map (invocation rest undefined p))))=_
    simp only [GridExecution.sequence]
    cases run : SharedIndices.one rest Native.sourceConfig (KernelModel.fallback undefined) p pid s with
    | none => rfl
    | some next => exact ih next

theorem result (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) (geometry : Geometry.Valid (ScaledWord.geometry p))
    (even : p.even=Bool.true → p.inner=p.steps*p.blockInner) (cover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output)
    (vals : (io rest undefined p).Inputs) (initial : State KernelModel.Word)
    (hin : (io rest undefined p).InputBinding vals initial) :
    ∃ final,runSchedule ((io rest undefined p).schedule vals) initial=some final ∧
      (io rest undefined p).OutputBinding vals (formula undefined p vals) final ∧
      (io rest undefined p).Frame vals initial final := by
  obtain ⟨final,executed,values,frame⟩ := SharedIndices.grid_correct rest Native.sourceConfig
    (KernelModel.fallback undefined) p geometry even cover ha hb hs initial
  refine ⟨final,(schedule_equal rest undefined p _ initial).trans executed,?_,?_⟩
  · intro o j _
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    have output := values (j.1,j.2.1) trivial
    change mapCell id (final.mem p.output
      (j.1.val*p.outputRowStride+j.2.1.val*p.outputColumnStride))=
      .mk .int (formula undefined p vals (0 : Fin 1) j)
    simp only [Geometry.address,ScaledWord.geometry] at output
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
theorem single_safe (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
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
    (initial : State KernelModel.Word) :
    TraceSafe (ScaledWord.backend rest Native.sourceConfig (KernelModel.fallback undefined))
      bounds (Kernels.scaledSharedIndices p) initial := by
  apply SharedIndices.safe
  · intro k hk i live
    have inner := active_inner p even k hk i.2.1 live
    have bound := ha (ScaledWord.inputRow p initial i.1) (k*p.blockInner+i.2.1.val)
      (Nat.mod_lt _ nonempty.1) inner
    simpa only [ScaledWord.aOffset,ScaledWord.inputInner,Nat.add_mul,Nat.mul_assoc,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using bound
  · intro k hk i live
    have inner := active_inner p even k hk i.1 live
    have bound := hb (k*p.blockInner+i.1.val) (ScaledWord.inputColumn p initial i.2.1)
      inner (Nat.mod_lt _ nonempty.2)
    simpa only [ScaledWord.bOffset,ScaledWord.inputInner,Nat.add_mul,Nat.mul_assoc,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using bound
  · intro i hi
    exact hs _ hi.1
  · intro i hi
    simpa only [ScaledWord.outputOffset,Nat.add_comm] using ho _ _ hi.1 hi.2

theorem schedule_safe (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
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
    (initial : State KernelModel.Word) :
    ScheduleSafe bounds (pids.map (invocation rest undefined p)) initial := by
  induction pids generalizing initial with
  | nil => trivial
  | cons pid pids ih =>
    refine ⟨?_,fun next _ => ih (fun q hq => reached q (by simp [hq])) next⟩
    exact single_safe rest undefined p even
      (reached_nonempty p geometry pid (reached pid (by simp))) bounds ha hb hs ho
      (GridExecution.atProgram initial pid)

theorem complete (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) (geometry : Geometry.Valid (ScaledWord.geometry p))
    (even : p.even=Bool.true → p.inner=p.steps*p.blockInner) (cover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) :
    (io rest undefined p).Implements (formula undefined p) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hbound initial _
    by_cases live : 0<p.rows ∧ 0<p.columns
    · apply schedule_safe rest undefined p geometry even bounds
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
      change ScheduleSafe bounds ((List.range (Geometry.count (ScaledWord.geometry p))).map (invocation rest undefined p)) initial
      rw [zero]
      trivial
  · exact result rest undefined p geometry even cover ha hb hs
end CandidateScaledIO

end
end VeriTile.Bench.Repairs.IntScaledMatmul
