import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchShape
import bench.tritonbench_g.lora_expand_gemv.Proofs.HostMetadata

/-! The repaired Python kernel is byte-for-byte unchanged. This module reuses
its complete typed source and gives its arithmetic a symbolic interpretation.
A reduction is a valid explicit tree, never a callback returning a dot product.
Native contraction is interpreted later; cfg does not evaluate symbols. -/
namespace VeriTile.Bench.Repairs.LoraExpandGemv
open VeriTile.Triton FP.Structural
noncomputable section
namespace FPValue
abbrev OutputFormat := TritonBenchG.LoraExpandGemv.FPValue.OutputFormat
open FP.Equational (ReductionTree ReductionPlan)

inductive Word where
  | zero
  | leaf (region : RegionName) (offset : Nat)
  | cast (src dst : OutputFormat) (x : Word)
  | mul (fmt : OutputFormat) (x y : Word)
  | add (fmt : OutputFormat) (x y : Word)
  deriving DecidableEq, Inhabited

/-- Identity casts and the exact conversion of positive zero are syntactically
normalized. Nonzero conversions remain explicit boundaries to FMA contraction. -/
def castWord (_cfg : FP.Config) (src dst : OutputFormat) (x : Word) : Word :=
  if src=dst then x else match x with | .zero => .zero | _ => .cast src dst x

def mulWord (_cfg : FP.Config) := Word.mul
def addWord (_cfg : FP.Config) := Word.add

def treeValue (cfg : FP.Config) (fmt : OutputFormat) (xs : Fin n → Word) : ReductionTree n → Word
  | .input i => xs i
  | .zero => .zero
  | .add a b => addWord cfg fmt (treeValue cfg fmt xs a) (treeValue cfg fmt xs b)

def sumValues (cfg : FP.Config) (fmt : OutputFormat) (plans : ∀ n, ReductionPlan n)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Word .real shape) :
    Values Word .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => treeValue cfg fmt (fun i => xs (TileShape.insertAxisIndex shape ax out i)) (plans _).tree
  · exact fun out => treeValue cfg fmt (fun i => xs (TileShape.replaceAxisIndex shape ax out i)) (plans _).tree

def formatOf : FloatDType → Option OutputFormat
  | .fp16 => some .half | .bf16 => some .bfloat | .fp32 | .real => some .single | _ => none

def typedSum (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (d : FloatDType)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Word d.toTileDType shape) :
    Option (Values Word d.toTileDType (TileShape.reduceShape shape ax keep)) :=
  match formatOf d with
  | none => none
  | some fmt => some (fun out => ofFloat d
      (sumValues cfg fmt plans ax keep (fun i => toFloat d (xs i)) out))

def backend (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) : Algebra Word :=
  { rest with
    literal := fun p d x => if x=0 then .zero else rest.literal p d x
    binary := fun p d op x y => match formatOf d,op with
      | some fmt,.mul => mulWord cfg fmt x y
      | some fmt,.add => addWord cfg fmt x y
      | _,_ => rest.binary p d op x y
    cast := fun p src dst x => match formatOf src,formatOf dst with
      | some a,some b => castWord cfg a b x
      | _,_ => rest.cast p src dst x
    reduceSumTyped := fun _ d => typedSum cfg plans d }

@[simp] theorem castWord_same (cfg : FP.Config) (fmt : OutputFormat) (x : Word) :
    castWord cfg fmt fmt x=x := by simp [castWord]
@[simp] theorem castWord_zero (cfg : FP.Config) (src dst : OutputFormat) :
    castWord cfg src dst .zero=.zero := by simp [castWord]

end FPValue
namespace Branches
abbrev InputProfile := TritonBenchG.LoraExpandGemv.Branches.InputProfile
abbrev Parameters := TritonBenchG.LoraExpandGemv.Branches.Parameters
abbrev length := TritonBenchG.LoraExpandGemv.Branches.length
abbrev inputFormat := TritonBenchG.LoraExpandGemv.Branches.inputFormat
abbrev weightFormat := TritonBenchG.LoraExpandGemv.Branches.weightFormat
abbrev effectiveFormat := TritonBenchG.LoraExpandGemv.Branches.effectiveFormat
abbrev promote := TritonBenchG.LoraExpandGemv.Branches.promote
abbrev productFormat := TritonBenchG.LoraExpandGemv.Branches.productFormat
abbrev resultFormat := TritonBenchG.LoraExpandGemv.Branches.resultFormat
abbrev source := TritonBenchG.LoraExpandGemv.Branches.source

theorem complete_source_identical (p : Parameters) :
    source .singleHalf .single p=TritonBenchG.LoraExpandGemv.Branches.source .singleHalf .single p := rfl
end Branches
#axiomsClean FPValue.castWord_same
#axiomsClean FPValue.castWord_zero
#axiomsClean Branches.complete_source_identical
end
end VeriTile.Bench.Repairs.LoraExpandGemv
