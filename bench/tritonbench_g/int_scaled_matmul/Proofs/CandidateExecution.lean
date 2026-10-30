import bench.tritonbench_g.int_scaled_matmul.Proofs.CandidateUnscaled
import bench.tritonbench_g.int_scaled_matmul.Proofs.SourcePreservation
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.IntScaledMatmul
open VeriTile.Triton FP.Structural Revision001
open GridExecution (Outside atProgram)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
local instance : Inhabited KernelModel.Word := ⟨FP.Value.zero .fp32⟩
namespace SharedIndices
open ScaledWord
private theorem min_lt_form (a b : Nat) : min a b=if a<b then a else b := by
  by_cases h : a<b
  · rw [if_pos h,min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h,min_eq_right (Nat.le_of_not_gt h)]
attribute [local irreducible] FP.mul FP.round FP.IntegerConversion.toSigned FP.IntegerConversion.totalSigned
/-- The loop preserves the row/column indices already computed by the prelude. -/
structure Indices (p : Parameters) (initial current : State Word) : Prop where
  rows : current.regs .nat [p.blockRows] "rm" =
    some (fun i => rowBlock p initial*p.blockRows+i.1.val)
  columns : current.regs .nat [p.blockColumns] "rn" =
    some (fun i => columnBlock p initial*p.blockColumns+i.1.val)

theorem prelude_indices (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial : State Word) :
    Indices p initial (preludeExecution rest cfg u p initial).val := by
  constructor <;> simp [preludeExecution,blocksM,blocksN,firstRow,rowsInGroup,rowBlock,columnBlock,min_lt_form,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals funext i
  all_goals rfl


theorem loop (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word) (k : Nat) (hk : k≤p.steps)
    (h : Invariant p initial k current) (indices : Indices p initial current) :
    ∃ final,FP.Structural.range (backend rest cfg u) "j" k p.steps 1 (body p) current=some final ∧
      Invariant p initial p.steps final ∧ Indices p initial final := by
  have aux : ∀ n k (current : State Word),p.steps-k=n → k≤p.steps →
      Invariant p initial k current → Indices p initial current →
      ∃ final,FP.Structural.range (backend rest cfg u) "j" k p.steps 1 (body p) current=some final ∧
        Invariant p initial p.steps final ∧ Indices p initial final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h indices
      by_cases live : k<p.steps
      · let next := bodyExecution rest cfg u p initial current k h
        have ni : Indices p initial next.val := by
          constructor <;> simp [next,bodyExecution,bodyResult,indices.rows,indices.columns]
        obtain ⟨final,ran,hi,fi⟩ := ih (p.steps-(k+1)) (by omega) (k+1) next.val rfl (by omega)
          (bodyInvariant rest cfg u p initial current k h) ni
        refine ⟨final,?_,hi,fi⟩
        rw [FP.Structural.range,if_neg (by decide),if_pos live,next.property]
        exact ran
      · have eq : k=p.steps := by omega
        subst k
        refine ⟨current,?_,h,indices⟩
        rw [FP.Structural.range,if_neg (by decide),if_neg (by omega)]
  exact aux _ _ _ rfl hk h indices

/-- Proof-only split; the authoritative candidate is the direct triton block. -/
def epilogue (p : Parameters) := (ScaledWord.epilogue p).drop 2

theorem decomposition (p : Parameters) : (Kernels.scaledSharedIndices p).surfaceBody =
    prelude p ++ [.forRange "j" 0 p.steps 1 (body p)] ++ epilogue p := by rfl

def finish (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word)
    (h : Invariant p initial p.steps current) (indices : Indices p initial current) :
    {t : State Word // run (backend rest cfg u) (epilogue p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue, ScaledWord.epilogue, original, typedScaled, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    FP.Structural.bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    h.pidm, h.pidn, h.acc, indices.rows, indices.columns, h.memory, h.undefined, backend,
    FP.IntegerConversion.fp32TotalBackend, FP.IntegerConversion.fp32Source,
    FP.Software.scalarFP32, FP.Software.fp32Context]
  rfl


theorem finish_memory (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word)
    (h : Invariant p initial p.steps current) (indices : Indices p initial current) :
    (finish rest cfg u p initial current h indices).val.mem = (TotalScaled.written cfg u p initial).mem := by
  simp +contextual [finish, TotalScaled.written, masked_scatter_memory, h.memory,
    active, outputOffset, outputRow, outputColumn, TotalScaled.expected, scaled,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm]


theorem executionMemory (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial : State Word) :
    ∃ final, FP.Structural.exec (backend rest cfg u) (Kernels.scaledSharedIndices p) initial = some final ∧
      final.mem = (TotalScaled.written cfg u p initial).mem := by
  let start := preludeExecution rest cfg u p initial
  obtain ⟨mid, hm, hi, indices⟩ := loop rest cfg u p initial start.val 0 (by omega)
    (preludeInvariant rest cfg u p initial) (prelude_indices rest cfg u p initial)
  let finish := finish rest cfg u p initial mid hi indices
  refine ⟨finish.val, ?_, finish_memory rest cfg u p initial mid hi indices⟩
  change run (backend rest cfg u) (Kernels.scaledSharedIndices p).surfaceBody initial = some finish.val
  rw [show (Kernels.scaledSharedIndices p).surfaceBody = prelude p ++ [.forRange "j" 0 p.steps 1 (body p)] ++ epilogue p from decomposition p,
    run_append, run_append, start.property]
  have hsingle (s : State Word) :
      run (backend rest cfg u) [.forRange "j" 0 p.steps 1 (body p)] s =
        FP.Structural.range (backend rest cfg u) "j" 0 p.steps 1 (body p) s := by simp [run,step]
  simp only [Option.bind_some, hsingle, hm]
  exact finish.property

theorem correctness (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial : State Word)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend rest cfg u) (Kernels.scaledSharedIndices p) initial = some final ∧
      Post p initial final (TotalScaled.expected cfg u p initial) := by
  obtain ⟨final,he,hm⟩ := executionMemory rest cfg u p initial
  refine ⟨final,he,?_,?_⟩
  · intro i hi
    rw [hm]
    exact masked_readback_active _ _ _ _ _ i hi hw
  · intro r o hframe
    rw [hm]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (hframe.imp id (fun h i _ => h i)) _

private theorem singleton_safe (M : Algebra Word) (bounds : RegionName → Nat)
    (st : ComputeStmt) (s : State Word) :
    Safety.traceSafeList M bounds [st] s ↔ Safety.stepSafe M bounds st s := by
  rw [Safety.traceSafeList_cons]
  simp
theorem epilogue_safe (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (bounds : RegionName → Nat) (initial current : State Word)
    (h : Invariant p initial p.steps current) (indices : Indices p initial current)
    (hs : ∀ i : CIndex p,active p initial i → outputRow p initial i*p.scaleRowStride<bounds p.scale)
    (ho : ∀ i : CIndex p,active p initial i → outputOffset p initial i<bounds p.output) :
    Safety.traceSafeList (backend rest cfg u) bounds (epilogue p) current := by
  simp [epilogue,ScaledWord.epilogue,original,typedScaled,ComputeKernel.surfaceBody,
    singleton_safe,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,Safety.precisionContext,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex,
    h.pidm,h.pidn,h.acc,indices.rows,indices.columns,backend]
  constructor
  · intro r c hr hc
    exact hs (r,c,PUnit.unit) ⟨hr,hc⟩
  · intro r c hr hc
    exact ho (r,c,PUnit.unit) ⟨hr,hc⟩

theorem safe (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (bounds : RegionName → Nat) (initial : State Word)
    (ha : ∀ k,k<p.steps → ∀ i : AIndex p,
      p.even=Bool.true ∨ inputInner p initial i.2.1<p.inner-k*p.blockInner →
      aOffset p initial k i<bounds p.input)
    (hb : ∀ k,k<p.steps → ∀ i : BIndex p,
      p.even=Bool.true ∨ inputInner p initial i.1<p.inner-k*p.blockInner →
      bOffset p initial k i<bounds p.weight)
    (hs : ∀ i : CIndex p,active p initial i → outputRow p initial i*p.scaleRowStride<bounds p.scale)
    (ho : ∀ i : CIndex p,active p initial i → outputOffset p initial i<bounds p.output) :
    TraceSafe (backend rest cfg u) bounds (Kernels.scaledSharedIndices p) initial := by
  let start := preludeExecution rest cfg u p initial
  have startInv := preludeInvariant rest cfg u p initial
  obtain ⟨mid,executed,midInv,indices⟩ := loop rest cfg u p initial start.val 0 (by omega) startInv
    (prelude_indices rest cfg u p initial)
  change Safety.traceSafeList (backend rest cfg u) bounds (Kernels.scaledSharedIndices p).surfaceBody initial
  rw [decomposition]
  rw [Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · rw [Safety.traceSafeList_append]
    refine ⟨ScaledSafety.prelude_safe rest cfg u p bounds initial,?_⟩
    intro s hs
    rw [start.property] at hs
    cases hs
    rw [singleton_safe,Safety.stepSafe]
    exact ScaledSafety.loop_safe rest cfg u p bounds initial start.val 0 (by omega) startInv ha hb
  · intro final hf
    rw [run_append,start.property] at hf
    simp only [Option.bind_some,run,step,executed] at hf
    cases hf
    exact epilogue_safe rest cfg u p bounds initial mid midInv indices hs ho

def one (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (pid : Nat) (s : State Word) :=
  exec (backend rest cfg u) (Kernels.scaledSharedIndices p) (atProgram s pid)
def grid (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (s : State Word) :=
  GridExecution.sequence (one rest cfg u p) (List.range (Geometry.count (geometry p))) s
theorem single_logical (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters)
    (hgeo : Geometry.Valid (geometry p))
    (heven : p.even=Bool.true → p.inner=p.steps*p.blockInner) (hcover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hscale : p.scale≠p.output)
    (initial : State Word) (current : State Word) (hout : Outside p.output initial current) (pid : Nat) :
    ∃ final, one rest cfg u p pid current=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : LogicalIndex p => Cell.mk .int (TotalScaled.logicalResult cfg u p initial x.1.val x.2.val))
        (Geometry.Owns (geometry p) pid) current final := by
  let s := atProgram current pid
  have hw : ∀ i j, active p s i → active p s j → outputOffset p s i=outputOffset p s j → i=j := by
    intro i j hi hj he
    apply Geometry.offset_injective (geometry p) hgeo pid i j hi hj
    simpa only [s,offset_at] using he
  obtain ⟨final,he,hread,hframe⟩ := correctness rest cfg u p s hw
  refine ⟨final,he,?_,?_⟩
  · intro x hx
    obtain ⟨i,hr,hc⟩ := hx
    have hi : active p s i := Geometry.owns_active (geometry p) pid x i hr hc
    have ho : outputOffset p s i=Geometry.address (geometry p) x := by
      rw [show s=atProgram current pid from rfl,offset_at]
      simp only [Geometry.offset,Geometry.address,hr,hc]
    rw [← ho,hread i hi,TotalScaled.expected_logical cfg u p s heven hcover i hi]
    change Cell.mk .int (TotalScaled.logicalResult cfg u p s (Geometry.outRow (geometry p) pid i)
      (Geometry.outColumn (geometry p) pid i))=_
    rw [hr,hc,TotalScaled.result_preserved cfg u p initial s ha hb hscale hout]
  · intro r o h
    apply hframe r o
    rcases h with h|h
    · exact Or.inl h
    · right
      intro i hi
      let x := Geometry.index (geometry p) pid i hi
      have hx : Geometry.Owns (geometry p) pid x := ⟨i,rfl,rfl⟩
      have ho : outputOffset p s i=Geometry.address (geometry p) x := by
        exact offset_at p current pid i
      rw [ho]
      exact h x hx

theorem grid_correct (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters)
    (hgeo : Geometry.Valid (geometry p))
    (heven : p.even=Bool.true → p.inner=p.steps*p.blockInner) (hcover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hscale : p.scale≠p.output)
    (initial : State Word) :
    ∃ final, grid rest cfg u p initial=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : LogicalIndex p => Cell.mk .int (TotalScaled.logicalResult cfg u p initial x.1.val x.2.val))
        (fun _ => True) initial final := by
  obtain ⟨final,he,hread,hframe⟩ := GridExecution.compose p.output (Geometry.address (geometry p))
    hgeo.writable (fun x : LogicalIndex p => Cell.mk .int (TotalScaled.logicalResult cfg u p initial x.1.val x.2.val))
    (Geometry.Owns (geometry p)) (one rest cfg u p) initial (Geometry.count (geometry p))
    (fun pid _ current hout => single_logical rest cfg u p hgeo heven hcover ha hb hscale initial current hout pid)
    (List.range (Geometry.count (geometry p))) (fun _ h => List.mem_range.mp h) initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro x _
    obtain ⟨pid,hpid,hx⟩ := Geometry.covers (geometry p) hgeo x
    exact hread x ⟨pid,List.mem_range.mpr hpid,hx⟩
  · intro r o h
    exact hframe r o (h.imp id (fun h x _ => h x trivial))

end SharedIndices
end
end VeriTile.Bench.Repairs.IntScaledMatmul
