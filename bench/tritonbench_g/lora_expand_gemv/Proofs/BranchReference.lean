import bench.tritonbench_g.lora_expand_gemv.Sources.Kernels

namespace VeriTile.Bench.Repairs.LoraExpandGemv.Branches
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat castWord mulWord addWord treeValue)
open FP.Equational (ReductionPlan)
noncomputable section
attribute [local simp] VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.fullSource VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.source VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.length VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.inputFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.weightFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.effectiveFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.promote VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.productFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.resultFormat
local instance : Inhabited Word := ⟨FPValue.Word.zero⟩

/-- Static float32 pointer annotations use the historical erased real channel. -/
def dtype : OutputFormat → FloatDType
  | .half => .fp16 | .bfloat => .bf16 | .single => .real

def engine (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) : Algebra Word :=
  { FPValue.backend rest cfg plans with
    fp32Load := id
    reduceSum := fun _ => FPValue.sumValues cfg .single plans }

def selected (p : Parameters) (s : State Word) : Int :=
  (s.mem p.indices.cast (s.pids 1)).read .int

def aOffset (p : Parameters) (s : State Word) (k : Nat) : Nat :=
  s.pids 1*p.inputBatchStride+k*p.inputInnerStride

def weightBase (p : Parameters) (s : State Word) : Nat :=
  ((p.loraStride : Int) * selected p s +
    (s.pids 0 : Int)*(length p : Int)*(p.weightRowStride : Int)).toNat

def bOffset (p : Parameters) (s : State Word) (m k : Nat) : Nat :=
  weightBase p s+m*p.weightRowStride+k*p.weightInnerStride

def outputBase (p : Parameters) (s : State Word) : Nat :=
  s.pids 1*p.outputBatchStride+s.pids 0*length p

def outputOffset (p : Parameters) (s : State Word) (m : Nat) : Nat :=
  outputBase p s+m*p.outputStride

def aLoaded (profile : InputProfile) (p : Parameters) (s : State Word) (k : Fin p.blockInner) : Word :=
  if p.evenK || decide (k.val < p.inner) then
    toFloat (dtype (inputFormat profile))
      ((s.mem p.input (aOffset p s k.val)).read (dtype (inputFormat profile)).toTileDType)
  else FPValue.Word.zero

def aValue (cfg : FP.Config) (profile : InputProfile) (p : Parameters) (s : State Word)
    (k : Fin p.blockInner) : Word :=
  if p.castType then castWord cfg (inputFormat profile) (weightFormat profile) (aLoaded profile p s k)
  else aLoaded profile p s k

def bValue (profile : InputProfile) (p : Parameters) (initial : State Word)
    (mem : RegionName → Nat → Cell Word) (m : Nat) (k : Fin p.blockInner) : Word :=
  if m < length p ∧ k.val < p.inner then
    toFloat (dtype (weightFormat profile))
      ((mem p.weight (bOffset p initial m k.val)).read (dtype (weightFormat profile)).toTileDType)
  else FPValue.Word.zero

def product (cfg : FP.Config) (profile : InputProfile) (p : Parameters) (initial : State Word)
    (mem : RegionName → Nat → Cell Word) (m : Nat) (k : Fin p.blockInner) : Word :=
  mulWord cfg (productFormat profile p)
    (castWord cfg (effectiveFormat profile p) (productFormat profile p) (aValue cfg profile p initial k))
    (castWord cfg (weightFormat profile) (productFormat profile p) (bValue profile p initial mem m k))

def accumulated (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (p : Parameters) (initial : State Word) (mem : RegionName → Nat → Cell Word) (m : Nat) : Word :=
  treeValue cfg (productFormat profile p) (product cfg profile p initial mem m) (plans p.blockInner).tree

def added (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial : State Word)
    (mem : RegionName → Nat → Cell Word) (m : Nat) : Word :=
  let acc := accumulated cfg plans profile p initial mem m
  if p.addInputs then
    addWord cfg (resultFormat profile out p)
      (castWord cfg (productFormat profile p) (resultFormat profile out p) acc)
      (castWord cfg out (resultFormat profile out p)
        (toFloat (dtype out) ((mem p.output (outputOffset p initial m)).read (dtype out).toTileDType)))
  else acc

def outputCell (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial : State Word)
    (mem : RegionName → Nat → Cell Word) (m : Nat) : Cell Word :=
  .mk (dtype out).toTileDType (ofFloat (dtype out)
    (castWord cfg (resultFormat profile out p) out (added cfg plans profile out p initial mem m)))

/-- Every lane reads the same block-entry memory before ordered writes.
Later blocks observe earlier stores, including any weight/output aliases. -/
def writeBlock (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial : State Word)
    (mem : RegionName → Nat → Cell Word) (n : Nat) : RegionName → Nat → Cell Word :=
  (TileShape.allIndices [p.blockRows]).foldl (fun outmem i => if n+i.1.val < length p then
    fun r o => if r=p.output ∧ o=outputOffset p initial (n+i.1.val) then
      outputCell cfg plans profile out p initial mem (n+i.1.val) else outmem r o
    else outmem) mem

def reference (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial : State Word) :
    Nat → Nat → (RegionName → Nat → Cell Word) → (RegionName → Nat → Cell Word)
  | 0, _, mem => mem
  | fuel+1, n, mem => if n < length p then
      reference cfg plans profile out p initial fuel (n+p.blockRows)
        (writeBlock cfg plans profile out p initial mem n) else mem

def expectedMemory (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial : State Word) : RegionName → Nat → Cell Word :=
  if selected p initial = -1 then initial.mem
  else reference cfg plans profile out p initial (length p) 0 initial.mem

def Outside (p : Parameters) (initial : State Word) (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.output ∨ ∀ m : Fin (length p), o ≠ outputOffset p initial m.val

structure Invariant (cfg : FP.Config) (profile : InputProfile)
    (p : Parameters) (initial current : State Word) : Prop where
  offk : current.regs .nat [p.blockInner] "offset_k" = some (fun i => i.1.val)
  offn : current.regs .nat [p.blockRows] "offset_n" = some (fun i => i.1.val)
  av : current.regs (dtype (effectiveFormat profile p)).toTileDType [p.blockInner] "tiled_a" =
    some (fun i => ofFloat (dtype (effectiveFormat profile p)) (aValue cfg profile p initial i.1))
  bp : current.regs .ptr [] "b_ptr" = some (fun _ => (p.weight,weightBase p initial))
  cp : current.regs .ptr [] "c_ptr" = some (fun _ => (p.output,outputBase p initial))
  size : current.regs .nat [] "split_n_length" = some (fun _ => length p)

#specNonCircular expectedMemory avoiding [source, FP.Structural.exec]
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Branches
