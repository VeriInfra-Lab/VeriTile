import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceParts
import bench.tritonbench_g.sgmv_expand_slice.Proofs.MaskedStore
import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceStoreArithmetic

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceSuffix
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

theorem common (abi : SourceABI) (p : KernelArgs) :
    SourceParts.suffix abi p = SourceParts.suffix
      {abi with input := .single, startsWidth := .w32, lengthsWidth := .w32, adaptersWidth := .w32} p := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases starts <;> cases lengths <;> cases adapters <;> rfl

def storageType : ScalarFormat → FloatDType
  | .half => .fp16 | .bfloat => .bf16 | .single => .real

def castIfNeeded {α : Type} (M : Algebra α) (precision : Option ComputeDType)
    (source target : FloatDType) (x : α) :=
  if source = target then x else M.cast precision source target x

def storePrecision (abi : SourceABI) : Option ComputeDType :=
  if abi.output = .single then some .fp32 else none

def additionType (abi : SourceABI) : FloatDType := match abi.output,abi.weight with
  | .single,_ => .real
  | .half,_ => .fp16
  | .bfloat,.half => .fp16
  | .bfloat,.bfloat => .bf16

def finish {α : Type} (M : Algebra α) (abi : SourceABI) (acc old : α) : α :=
  let rounded := M.cast none .real abi.weight.scalar.dtype acc
  if abi.addInputs then
    castIfNeeded M (storePrecision abi) (additionType abi) (storageType abi.output)
      (M.binary (storePrecision abi) (additionType abi) .add
        (castIfNeeded M (some .fp32) abi.weight.scalar.dtype (additionType abi) rounded)
        (castIfNeeded M (some .fp32) (storageType abi.output) (additionType abi) old))
  else castIfNeeded M (storePrecision abi) abi.weight.scalar.dtype (storageType abi.output) rounded

structure Coordinates where
  (start length tileRow tileColumn : Nat)

def row (c : Coordinates) (i : TileIndex [32,32]) := c.start+i.1.val+c.tileRow*32
def column (p : KernelArgs) (c : Coordinates) (i : TileIndex [32,32]) := i.2.1.val+c.tileColumn*32+p.slice
def active (p : KernelArgs) (c : Coordinates) (i : TileIndex [32,32]) : Prop :=
  row c i < c.start+c.length ∧ column p c i < p.slice+p.columns
instance (p : KernelArgs) (c : Coordinates) : DecidablePred (active p c) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))
def offset (p : KernelArgs) (c : Coordinates) (i : TileIndex [32,32]) :=
  (if row c i < c.start+c.length then row c i else 0)*p.outputRowStride +
  (if column p c i < p.slice+p.columns then column p c i else p.slice)*p.outputColumnStride

def outputValues {α : Type} [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (c : Coordinates) (s : State α) (acc : Values α .real [32,32]) :=
  fun i =>
    let d := storageType abi.output
    let old := if active p c i then toFloat d ((s.mem p.output (offset p c i)).read d.toTileDType)
      else toFloat d (s.undef d.toTileDType p.output (offset p c i))
    finish M abi (acc i) (if abi.output = .single then M.fp32Load old else old)

def written {α : Type} [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (c : Coordinates) (s : State α) (acc : Values α .real [32,32]) :=
  (TileShape.allIndices [32,32]).foldl
    (fun t i => if active p c i then t.write p.output (offset p c i)
      (.mk (storageType abi.output).toTileDType
        (ofFloat (storageType abi.output) (outputValues M abi p c s acc i))) else t) s

structure Ready {α : Type} (p : KernelArgs) (c : Coordinates)
    (s t : State α) (acc : Values α .real [32,32]) : Prop where
  mem : t.mem = s.mem
  undef : t.undef = s.undef
  start : t.regs .int [] "cur_seq_start" = some (fun _ => (c.start : Int))
  row : t.regs .int [] "pid_m" = some (fun _ => (c.tileRow : Int))
  column : t.regs .int [] "pid_n" = some (fun _ => (c.tileColumn : Int))
  length : t.regs .int [] "M" = some (fun _ => (c.length : Int))
  columns : t.regs .int [] "N" = some (fun _ => (p.columns : Int))
  batch : t.regs .int [] "cur_batch" = some (fun _ => (p.batch : Int))
  rowStride : t.regs .int [] "cm_stride" = some (fun _ => (p.outputRowStride : Int))
  columnStride : t.regs .int [] "cn_stride" = some (fun _ => (p.outputColumnStride : Int))
  slice : t.regs .int [] "slice_offset" = some (fun _ => (p.slice : Int))
  acc : t.regs .real [32,32] "accumulator" = some acc

theorem suffix_run {α : Type} [Inhabited α] (M : Algebra α)
    (abi : SourceABI) (p : KernelArgs) (c : Coordinates) (s t : State α)
    (acc : Values α .real [32,32]) (h : Ready p c s t acc)
    (hlength : (s.mem p.lengths p.batch).read .int = (c.length : Int)) :
    ∃ final, run M (SourceParts.suffix abi p) t = some final ∧
      final.mem = (written M abi p c s acc).mem := by
  rw [common]
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases add <;>
    simp [SourceParts.suffix,SourceParts.active,SourceParts.afterTile,SourceParts.afterLength,
      SourceParts.thenBody,SourceParts.outer,kernel,sourceFace,sourceHalfSingle323232,
      sourceHalfHalf323232,sourceHalfBfloat323232,sourceBfloatSingle323232,
      sourceBfloatHalf323232,sourceBfloatBfloat323232,ComputeKernel.surfaceBody,
      ScalarFormat.dtype,LowFormat.scalar,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,bop,numericLt,Region.cast,ofFloat,toFloat,
      State.setReg,store,h.mem,h.undef,h.start,h.row,h.column,h.length,h.columns,h.batch,
      h.rowStride,h.columnStride,h.slice,h.acc,hlength]
  all_goals
    unfold written
    apply List.foldl_rel (r := fun (a b : State α) => a.mem = b.mem)
    · rfl
    · intro i _ u v huv
      rcases i with ⟨ir,ic,iu⟩
      cases iu
      simp only [SourceStoreArithmetic.output_offset]
      simp only [SourceStoreArithmetic.row_lt,SourceStoreArithmetic.column_lt]
      by_cases hi : active p c (ir,ic,PUnit.unit)
      all_goals
        simp only [active,row,column] at hi
        simp [active,row,column,offset,outputValues,finish,additionType,castIfNeeded,storePrecision,storageType,
          ScalarFormat.dtype,LowFormat.scalar,ofFloat,toFloat,hi,State.write,huv]

#axiomsClean common
#axiomsClean suffix_run
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceSuffix
