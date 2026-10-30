import bench.tritonbench_g.int_scaled_matmul.Proofs.IOResults
import bench.tritonbench_g.int_scaled_matmul.Proofs.BlockSafety
import bench.tritonbench_g.int_scaled_matmul.Proofs.ScaledSafety

namespace VeriTile.Bench.Repairs.IntScaledMatmul
open VeriTile.Triton FP.Structural Revision001
noncomputable section
local instance : Inhabited KernelModel.Word := ⟨FP.Value.zero .fp32⟩

/-- Empty output grids do not require backing for either read-only matrix or
for the optional row scale. This is the original early-return domain. -/
theorem empty_count (g : Geometry.Layout) (h : Geometry.Valid g)
    (empty : ¬(0<g.rows ∧ 0<g.columns)) : Geometry.count g=0 := by
  by_cases hr : 0<g.rows
  · have zero : Geometry.gridColumns g=0 := by
      apply Nat.div_eq_of_lt
      change g.columns+g.blockColumns-1<g.blockColumns
      have hn := h.blockColumns
      omega
    simp only [Geometry.count,zero,Nat.mul_zero]
  · have zero : Geometry.gridRows g=0 := by
      apply Nat.div_eq_of_lt
      change g.rows+g.blockRows-1<g.blockRows
      have hm := h.blockRows
      omega
    simp only [Geometry.count,zero,Nat.zero_mul]

namespace IntIO
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
    exact ⟨BlockSafety.complete rest p positive bounds ha hb ho (GridExecution.atProgram initial pid),
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
end IntIO

namespace ScaledIO
/-- A reached grid job entails nonempty logical axes. Empty matrices retain
an empty schedule, with no new nonempty-input condition on the public theorem. -/
theorem reached_nonempty (p : ScaledWord.Parameters)
    (geometry : Geometry.Valid (ScaledWord.geometry p)) (pid : Nat)
    (live : pid<Geometry.count (ScaledWord.geometry p)) : 0<p.rows ∧ 0<p.columns := by
  have hm := geometry.blockRows
  have hn := geometry.blockColumns
  change 0<p.blockRows at hm
  change 0<p.blockColumns at hn
  constructor
  · by_contra no
    have zero : Geometry.gridRows (ScaledWord.geometry p)=0 := by
      apply Nat.div_eq_of_lt
      change p.rows+p.blockRows-1<p.blockRows
      omega
    simp only [Geometry.count,zero,Nat.zero_mul] at live
    omega
  · by_contra no
    have zero : Geometry.gridColumns (ScaledWord.geometry p)=0 := by
      apply Nat.div_eq_of_lt
      change p.columns+p.blockColumns-1<p.blockColumns
      omega
    simp only [Geometry.count,zero,Nat.mul_zero] at live
    omega

theorem active_inner (p : ScaledWord.Parameters)
    (even : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (k : Nat) (hk : k<p.steps) (t : Fin p.blockInner)
    (live : p.even=Bool.true ∨ t.val<p.inner-k*p.blockInner) : k*p.blockInner+t.val<p.inner := by
  rcases live with live | live
  · have limit := Nat.mul_le_mul_right p.blockInner (Nat.succ_le_of_lt hk)
    rw [Nat.succ_mul,← even live] at limit
    have ht := t.isLt
    omega
  · omega

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
      bounds (ScaledWord.source p) initial := by
  apply ScaledSafety.complete
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
end ScaledIO

#axiomsClean IntIO.complete
#axiomsClean ScaledIO.complete
end
end VeriTile.Bench.Repairs.IntScaledMatmul
