import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.IOComplete
import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.SourcePreservation
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.IntScaledMatmul
open VeriTile.Triton FP.Structural Revision001
open GridExecution (Outside atProgram)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
local instance : Inhabited KernelModel.Word := ⟨FP.Value.zero .fp32⟩
namespace CopyFree
open BlockWord
/-- Proof-only decomposition of the full readable candidate. -/
def epilogue (p : Parameters) : List ComputeStmt :=
  [.assign .blockPtr [p.blockRows,p.blockColumns] "c_block_ptr"
    (.alg (.makeBlockPtrDynOffsets p.output (.constNat 0)
      [p.rows,p.columns] [p.blockRows,p.blockColumns]
      [p.outputRowStride,p.outputColumnStride]
      [.mul .nat Broadcast.nil (.ref .nat [] "pid_m") (.constNat p.blockRows),
       .mul .nat Broadcast.nil (.ref .nat [] "pid_n") (.constNat p.blockColumns)])),
   .store .int [p.blockRows,p.blockColumns]
     (.blockPtr (.ref .blockPtr [p.blockRows,p.blockColumns] "c_block_ptr") [0,1])
     (.alg (.ref .int [p.blockRows,p.blockColumns] "accumulator")) .none]

theorem decomposition (p : Parameters) : (Kernels.unscaledCopyFree p).surfaceBody =
    prelude p ++ [.forRange "k" 0 p.inner p.blockInner (body p)] ++ epilogue p := by rfl
def finish {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial current : State α)
    (h : Invariant p initial (steps p) current) :
    {t : State α // run (backend M) (epilogue p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    FP.Structural.bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    h.pidm, h.pidn, h.acc, h.memory, backend]
  rfl


theorem finish_memory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial current : State α)
    (h : Invariant p initial (steps p) current) :
    (finish M p initial current h).val.mem = (written M p initial).mem := by
  simp +contextual [finish, written, masked_scatter_memory, h.memory,
    active, outputOffset, outputValue, backend, BlockPtr.inBounds_2d_offsets, BlockPtr.address_2d_offsets,
    TileShape.indexToList,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm, Nat.mul_comm]


theorem correctness (rest : Algebra KernelModel.Word) (p : Parameters)
    (positive : 0<p.blockInner) (initial : State KernelModel.Word)
    (hw : ∀ i j,active p initial i → active p initial j →
      outputOffset p initial i=outputOffset p initial j → i=j) :
    ∃ final,exec (backend rest) (Kernels.unscaledCopyFree p) initial=some final ∧
      Post p initial final (outputValue p initial) := by
  let start := preludeExecution rest p initial
  obtain ⟨mid,ran,hi⟩ := loopExecution rest p positive initial start.val 0 (by omega)
    (preludeInvariant rest p initial)
  simp only [Nat.zero_mul] at ran
  let last := finish rest p initial mid hi
  refine ⟨last.val,?_,?_,?_⟩
  · change run (backend rest) (Kernels.unscaledCopyFree p).surfaceBody initial=some last.val
    rw [decomposition,run_append,run_append,start.property]
    simp only [Option.bind_some,run,step,ran]
    exact last.property
  · intro i active
    rw [finish_memory]
    exact VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_readback_active _ _ _ _ _ i active hw
  · intro r o frame
    rw [finish_memory]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (frame.imp id (fun h i _=>h i)) _

private theorem singleton_safe (M : Algebra KernelModel.Word) (bounds : RegionName → Nat)
    (st : ComputeStmt) (s : State KernelModel.Word) :
    Safety.traceSafeList M bounds [st] s ↔ Safety.stepSafe M bounds st s := by
  rw [Safety.traceSafeList_cons]
  simp

private theorem safe_block_pair (M : Algebra KernelModel.Word)
    (precision : Option ComputeDType) (bounds : RegionName → Nat)
    (s : State KernelModel.Word) (r : RegionName)
    (base : Op .nat []) (parent : List Nat) (shape : TileShape) (strides : List Nat)
    (row col : Op .nat []) :
    Safety.safeOp M precision bounds s (.makeBlockPtrDynOffsets r base parent shape strides [row,col]) ↔
      Safety.safeOp M precision bounds s base ∧ Safety.safeOp M precision bounds s row ∧
        Safety.safeOp M precision bounds s col := by
  rw [Safety.safeOp_unfold]
  simp

attribute [local simp 2000] safe_block_pair

theorem epilogue_safe (M : Algebra KernelModel.Word) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State KernelModel.Word)
    (h : Invariant p initial (steps p) current)
    (ho : ∀ r c,r<p.rows → c<p.columns →
      r*p.outputRowStride+c*p.outputColumnStride<bounds p.output) :
    Safety.traceSafeList (backend M) bounds (epilogue p) current := by
  simp [epilogue,ComputeKernel.surfaceBody,
    singleton_safe,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,FP.Structural.step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,
    Option.bind,integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex,
    h.pidm,h.pidn,h.acc,backend,TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets]
  intro r c hr hc
  simpa only [Nat.mul_comm] using ho _ _ hr hc

theorem safe (rest : Algebra KernelModel.Word) (p : Parameters) (positive : 0<p.blockInner)
    (bounds : RegionName → Nat)
    (ha : ∀ r k,r<p.rows → k<p.inner → r*p.inputRowStride+k*p.inputInnerStride<bounds p.input)
    (hb : ∀ k c,k<p.inner → c<p.columns → k*p.weightInnerStride+c*p.weightColumnStride<bounds p.weight)
    (ho : ∀ r c,r<p.rows → c<p.columns → r*p.outputRowStride+c*p.outputColumnStride<bounds p.output)
    (initial : State KernelModel.Word) :
    TraceSafe (backend rest) bounds (Kernels.unscaledCopyFree p) initial := by
  let start := preludeExecution rest p initial
  have startInv := preludeInvariant rest p initial
  obtain ⟨mid,executed,midInv⟩ := loopExecution rest p positive initial start.val 0 (by omega) startInv
  simp only [Nat.zero_mul] at executed
  change Safety.traceSafeList (backend rest) bounds (Kernels.unscaledCopyFree p).surfaceBody initial
  rw [decomposition,Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · rw [Safety.traceSafeList_append]
    refine ⟨BlockSafety.prelude_safe rest p bounds initial,?_⟩
    intro s hs
    rw [start.property] at hs
    cases hs
    rw [singleton_safe,Safety.stepSafe]
    simpa only [Nat.zero_mul] using BlockSafety.loop_safe rest p positive bounds initial start.val 0
      (by omega) startInv ha hb
  · intro final hf
    rw [run_append,start.property] at hf
    simp only [Option.bind_some,run,step,executed] at hf
    cases hf
    exact epilogue_safe rest p bounds initial mid midInv ho
def one (rest : Algebra KernelModel.Word) (p : Parameters) (pid : Nat) (s : State KernelModel.Word) :=
  exec (backend rest) (Kernels.unscaledCopyFree p) (GridExecution.atProgram s pid)
def grid (rest : Algebra KernelModel.Word) (p : Parameters) (s : State KernelModel.Word) :=
  GridExecution.sequence (one rest p) (List.range (Geometry.count (geometry p))) s
theorem single_logical (M : Algebra KernelModel.Word) (p : Parameters)
    (hgeo : Geometry.Valid (geometry p)) (hBK : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (initial current : State KernelModel.Word) (hout : Outside p.output initial current) (pid : Nat) :
    ∃ final, one M p pid current=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : LogicalIndex p => Cell.mk .int (logicalDot p initial x.1.val x.2.val))
        (Geometry.Owns (geometry p) pid) current final := by
  let s := atProgram current pid
  have hw : ∀ i j, active p s i → active p s j → outputOffset p s i=outputOffset p s j → i=j := by
    intro i j hi hj he
    apply Geometry.offset_injective (geometry p) hgeo pid i j hi hj
    simpa only [s,offset_at] using he
  obtain ⟨final,he,hread,hframe⟩ := CopyFree.correctness M p hBK s hw
  refine ⟨final,he,?_,?_⟩
  · intro x hx
    obtain ⟨i,hr,hc⟩ := hx
    have hi : active p s i := Geometry.owns_active (geometry p) pid x i hr hc
    have ho : outputOffset p s i=Geometry.address (geometry p) x := by
      rw [show s=atProgram current pid from rfl,offset_at]
      simp only [Geometry.offset,Geometry.address,hr,hc]
    rw [← ho,hread i hi,output_logical p hBK s i hi]
    change Cell.mk .int (logicalDot p s (Geometry.outRow (geometry p) pid i)
      (Geometry.outColumn (geometry p) pid i))=_
    rw [hr,hc,logical_preserved p initial s ha hb hout]
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

theorem grid_correct (M : Algebra KernelModel.Word) (p : Parameters)
    (hgeo : Geometry.Valid (geometry p)) (hBK : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (initial : State KernelModel.Word) :
    ∃ final, grid M p initial=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : LogicalIndex p => Cell.mk .int (logicalDot p initial x.1.val x.2.val))
        (fun _ => True) initial final := by
  obtain ⟨final,he,hread,hframe⟩ := GridExecution.compose p.output (Geometry.address (geometry p))
    hgeo.writable (fun x : LogicalIndex p => Cell.mk .int (logicalDot p initial x.1.val x.2.val))
    (Geometry.Owns (geometry p)) (one M p) initial (Geometry.count (geometry p))
    (fun pid _ current hout => single_logical M p hgeo hBK ha hb initial current hout pid)
    (List.range (Geometry.count (geometry p))) (fun _ h => List.mem_range.mp h) initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro x _
    obtain ⟨pid,hpid,hx⟩ := Geometry.covers (geometry p) hgeo x
    exact hread x ⟨pid,List.mem_range.mpr hpid,hx⟩
  · intro r o h
    exact hframe r o (h.imp id (fun h x _ => h x trivial))

end CopyFree

end
end VeriTile.Bench.Repairs.IntScaledMatmul
