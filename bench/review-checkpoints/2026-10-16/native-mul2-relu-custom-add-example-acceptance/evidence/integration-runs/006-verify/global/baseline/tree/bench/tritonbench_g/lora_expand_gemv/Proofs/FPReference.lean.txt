import bench.tritonbench_g.lora_expand_gemv.Proofs.FPModel

namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def selected (p : Parameters) (s : State Word) : Nat :=
  (s.mem p.indices.cast (s.pids 1)).read .nat

def aOffset (p : Parameters) (s : State Word) (k : Nat) : Nat :=
  s.pids 1 * p.inputBatchStride + k * p.inputInnerStride

def weightBase (p : Parameters) (s : State Word) : Nat :=
  p.loraStride * selected p s + s.pids 0 * p.length * p.weightRowStride

def bOffset (p : Parameters) (s : State Word) (m k : Nat) : Nat :=
  weightBase p s + m * p.weightRowStride + k * p.weightInnerStride

def outputBase (p : Parameters) (s : State Word) : Nat :=
  s.pids 1 * p.outputBatchStride + s.pids 0 * p.length

def outputOffset (p : Parameters) (s : State Word) (m : Nat) : Nat :=
  outputBase p s + m * p.outputStride

def aValue (profile : InputProfile) (p : Parameters) (initial : State Word) (k : Fin p.blockInner) : Word :=
  if k.val < p.inner then
    toFloat (inputType profile) ((initial.mem p.input (aOffset p initial k.val)).read (inputType profile).toTileDType)
  else FP.Value.zero .fp32

def bValue (profile : InputProfile) (p : Parameters) (initial : State Word) (m : Nat) (k : Fin p.blockInner) : Word :=
  if m < p.length ∧ k.val < p.inner then
    toFloat (weightType profile) ((initial.mem p.weight (bOffset p initial m k.val)).read (weightType profile).toTileDType)
  else FP.Value.zero .fp32

/-- All BLOCK_K products remain, including padded zero multiplications.
Inputs are captured from initial memory even when output aliases input. -/
def product (cfg : FP.Config) (profile : InputProfile) (p : Parameters) (initial : State Word)
    (m : Nat) (k : Fin p.blockInner) : Word :=
  mulWord cfg (productFormat profile)
    (castWord cfg (inputFormat profile) (productFormat profile) (aValue profile p initial k))
    (castWord cfg (weightFormat profile) (productFormat profile) (bValue profile p initial m k))

theorem product_function (cfg : FP.Config) (profile : InputProfile) (p : Parameters)
    (initial : State Word) (m : Nat) :
    product cfg profile p initial m = fun k =>
      mulWord cfg (productFormat profile)
        (castWord cfg (inputFormat profile) (productFormat profile) (aValue profile p initial k))
        (castWord cfg (weightFormat profile) (productFormat profile) (bValue profile p initial m k)) := rfl

def accumulated (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (p : Parameters) (initial : State Word) (m : Nat) : Word :=
  treeValue cfg (productFormat profile) (product cfg profile p initial m) (plans p.blockInner).tree

def expected (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial : State Word) (m : Nat) : Word :=
  castWord cfg (productFormat profile) out (accumulated cfg plans profile p initial m)

def outputCell (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial : State Word) (m : Nat) : Cell Word :=
  .mk (outputType out).toTileDType (ofFloat (outputType out) (expected cfg plans profile out p initial m))

def writeBlock (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial current : State Word) (n : Nat) : State Word :=
  (TileShape.allIndices [p.blockRows]).foldl (fun t i => if n+i.1.val < p.length then
    t.write p.output (outputOffset p initial (n+i.1.val))
      (outputCell cfg plans profile out p initial (n+i.1.val)) else t) current

def Outside (p : Parameters) (initial : State Word) (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.output ∨ ∀ m : Fin p.length, o ≠ outputOffset p initial m.val

structure Invariant (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial : State Word) (n : Nat) (current : State Word) : Prop where
  offk : current.regs .nat [p.blockInner] "offset_k" = some (fun i => i.1.val)
  offn : current.regs .nat [p.blockRows] "offset_n" = some (fun i => i.1.val)
  av : current.regs (inputType profile).toTileDType [p.blockInner] "tiled_a" =
    some (fun i => ofFloat (inputType profile) (aValue profile p initial i.1))
  bp : current.regs .ptr [] "b_ptr" = some (fun _ => (p.weight,weightBase p initial))
  cp : current.regs .ptr [] "c_ptr" = some (fun _ => (p.output,outputBase p initial))
  weights : current.mem p.weight = initial.mem p.weight
  readback : ∀ m : Fin p.length, current.mem p.output (outputOffset p initial m.val) =
    if m.val < n then outputCell cfg plans profile out p initial m.val
    else initial.mem p.output (outputOffset p initial m.val)
  frame : ∀ r o, Outside p initial r o → current.mem r o = initial.mem r o

#specNonCircular expected avoiding [source, FP.Structural.exec]
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue
