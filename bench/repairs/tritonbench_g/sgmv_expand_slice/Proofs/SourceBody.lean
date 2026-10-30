import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceParts
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.RangeReuse
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceArithmetic

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceBody
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

/-- A structural total implementation of the local dotAcc operation. Numerical
instantiations must define the selected 32×32×16 case from their own primitives;
this interface is not an assumed whole-kernel result or a numerical agreement. -/
structure DotImplementation (α : Type) where
  apply : Option ComputeDType → (left right output : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → α) → (TileIndex (batch++[k,n]) → α) →
    (TileIndex (batch++[m,n]) → α) → (TileIndex (batch++[m,n]) → α)

def install {α : Type} (M : Algebra α) (dot : DotImplementation α) : Algebra α :=
  {M with
    dotAcc := fun precision left right output options {_} {_} {_} {_} a b c =>
      some (fun i => ofFloat output (dot.apply precision left right output options
        (fun j => toFloat left (a j)) (fun j => toFloat right (b j))
        (fun j => toFloat output (c j)) i))}

structure Coordinates where
  (start adapter : Nat)
  rows : Fin 32 → Nat
  columns : Fin 32 → Nat

def globalInner (block : Nat) (k : Fin 16) : Nat := block*16+k.val
def safeInner (p : KernelArgs) (block : Nat) (k : Fin 16) : Nat :=
  if globalInner block k < p.inner then globalInner block k else 0

def aOffset (p : KernelArgs) (coordinates : Coordinates) (block : Nat)
    (i : TileIndex [32,16]) : Nat :=
  coordinates.start*p.inputRowStride+coordinates.rows i.1*p.inputRowStride+
    safeInner p block i.2.1*p.inputInnerStride
def bOffset (p : KernelArgs) (coordinates : Coordinates) (block : Nat)
    (i : TileIndex [16,32]) : Nat :=
  p.adapterStride*coordinates.adapter+safeInner p block i.1*p.weightInnerStride+
    coordinates.columns i.2.1*p.weightColumnStride

def aValues {α : Type} [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (coordinates : Coordinates) (s : State α) (block : Nat) :
    TileIndex [32,16] → α := fun i =>
  let value := if p.inner % 16 = 0 ∨ globalInner block i.2.1 < p.inner then
      toFloat abi.input.dtype ((s.mem p.input (aOffset p coordinates block i)).read abi.input.dtype.toTileDType)
    else M.cast none .real abi.input.dtype (M.literal none .real 0)
  if abi.input = abi.weight.scalar then value
  else M.cast none abi.input.dtype abi.weight.scalar.dtype value

def bValues {α : Type} [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (coordinates : Coordinates) (s : State α) (block : Nat) :
    TileIndex [16,32] → α := fun i =>
  if p.inner % 16 = 0 ∨ globalInner block i.1 < p.inner then
    toFloat abi.weight.scalar.dtype ((s.mem p.weight (bOffset p coordinates block i)).read abi.weight.scalar.dtype.toTileDType)
  else M.cast none .real abi.weight.scalar.dtype (M.literal none .real 0)

def update {α : Type} [Inhabited α] (M : Algebra α) (dot : DotImplementation α)
    (abi : SourceABI) (p : KernelArgs) (coordinates : Coordinates) (s : State α)
    (block : Nat) (acc : Values α .real [32,32]) : Values α .real [32,32] :=
  dot.apply (some .fp32) abi.weight.scalar.dtype abi.weight.scalar.dtype .real {}
    (batch := []) (m := 32) (k := 16) (n := 32)
    (aValues M abi p coordinates s block) (bValues M abi p coordinates s block) acc

def writes : List RegName :=
  ["k","global_k","safe_k","a_ptr","b_ptr","tiled_a","tiled_b","accumulator"]

structure Ready {α : Type} (p : KernelArgs) (coordinates : Coordinates)
    (s t : State α) (acc : Values α .real [32,32]) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  counts : t.numPids = s.numPids
  undef : t.undef = s.undef
  preserved : ∀ d sh name, name ∉ writes → t.regs d sh name = s.regs d sh name
  inner : t.regs .int [] "K" = some (fun _ => (p.inner : Int))
  xm : t.regs .int [] "xm_stride" = some (fun _ => (p.inputRowStride : Int))
  xk : t.regs .int [] "xk_stride" = some (fun _ => (p.inputInnerStride : Int))
  wa : t.regs .int [] "l0_stride" = some (fun _ => (p.adapterStride : Int))
  wn : t.regs .int [] "lora_k_stride" = some (fun _ => (p.weightColumnStride : Int))
  wk : t.regs .int [] "lora_n_stride" = some (fun _ => (p.weightInnerStride : Int))
  start : t.regs .int [] "cur_seq_start" = some (fun _ => (coordinates.start : Int))
  adapter : t.regs .int [] "lora_index" = some (fun _ => (coordinates.adapter : Int))
  offk : t.regs .int [16] "offset_k" = some (fun i => (i.1.val : Int))
  rows : t.regs .int [32] "ram" = some (fun i => (coordinates.rows i.1 : Int))
  columns : t.regs .int [32] "rbn" = some (fun i => (coordinates.columns i.1 : Int))
  acc : t.regs .real [32,32] "accumulator" = some acc

theorem body_run {α : Type} [Inhabited α] (M : Algebra α) (dot : DotImplementation α)
    (abi : SourceABI) (p : KernelArgs) (coordinates : Coordinates) (s t : State α)
    (block : Nat) (acc : Values α .real [32,32]) (h : Ready p coordinates s t acc) :
    ∃ final, run (install M dot) (SourceParts.body abi p)
      (t.setReg "k" .int [] (fun _ => (block : Int))) = some final ∧
      Ready p coordinates s final (update M dot abi p coordinates s block acc) := by
  rw [SourceParts.body_same_ABI]
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases input <;> cases weight <;>
    cases he : (p.inner % 16 == 0) <;>
    simp [SourceParts.body,SourceParts.loopParts,SourceParts.active,SourceParts.afterTile,
      SourceParts.afterLength,SourceParts.thenBody,SourceParts.outer,Source.intRange?,
      kernel,sourceFace,sourceHalfSingle323232,sourceBfloatSingle323232,
      ScalarFormat.dtype,LowFormat.scalar,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,
      natLt,numericLt,integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,install,
      State.setReg,he,h.inner,h.xm,h.xk,h.wa,h.wn,h.wk,h.start,h.adapter,h.offk,
      h.rows,h.columns,h.acc,h.mem]
  all_goals constructor
  all_goals first | rfl | exact h.pids | exact h.counts | exact h.undef |
    exact h.inner | exact h.xm | exact h.xk | exact h.wa | exact h.wn | exact h.wk |
    exact h.start | exact h.adapter | exact h.offk | exact h.rows | exact h.columns | skip
  all_goals try
    intro d sh name hn
    have hn' := hn
    simp only [writes,List.mem_cons,List.not_mem_nil,not_or] at hn'
    simpa [hn'] using h.preserved d sh name hn
  all_goals
    change some _ = some _
    apply congrArg some
    simp at he
    unfold update
    funext i
    apply congrArg₂ (fun (a : TileIndex [32,16] → α) (b : TileIndex [16,32] → α) =>
      dot.apply (some .fp32) _ _ .real {} (batch := []) (m := 32) (k := 16) (n := 32)
        a b acc i)
  all_goals
    funext i
    simp only [SourceArithmetic.input_offset,SourceArithmetic.weight_offset]
    simp [aValues,bValues,aOffset,bOffset,safeInner,globalInner,
      ScalarFormat.dtype,LowFormat.scalar,ofFloat,toFloat,he,
      SourceArithmetic.inner_lt]

#axiomsClean body_run

end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceBody
