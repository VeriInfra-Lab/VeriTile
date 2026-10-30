import bench.tritonbench_g.destindex_copy.Kernels
import bench.tritonbench_g.destindex_copy_kv1.Contract

namespace VeriTile.Bench.Repairs.DestindexCopy
noncomputable section
inductive Role where | nopeSource | ropeSource | destinations | nopeOutput | ropeOutput
  deriving DecidableEq, Repr
abbrev Inputs := Role → VeriTile.Torch.TensorMetadata
abbrev Word := DestindexCopyKv1.Word
abbrev Memory := DestindexCopyKv1.Memory
abbrev addr := DestindexCopyKv1.addr
abbrev destination := DestindexCopyKv1.destination

def plane (m : Inputs) : Side → DestindexCopyKv1.Inputs
  | .nope, .source => m .nopeSource
  | .rope, .source => m .ropeSource
  | _, .destinations => m .destinations
  | .nope, .output => m .nopeOutput
  | .rope, .output => m .ropeOutput

def config (m : Inputs) (s : Side) : Config := DestindexCopyKv1.config (plane m s)
def base (m : Inputs) (s : Side) (r : DestindexCopyKv1.Role) : Nat := DestindexCopyKv1.base (plane m s) r

def SpanSeparate (a : Nat) (sizeA : Nat) (b : Nat) (sizeB : Nat) : Prop :=
  a+4*sizeA≤b ∨ b+4*sizeB≤a

def Separated (m : Inputs) : Prop := ∀ s t,
  SpanSeparate (base m s .output) ((plane m s) .output).numel
    (base m t .source) ((plane m t) .source).numel ∧
  SpanSeparate (base m s .output) ((plane m s) .output).numel
    (base m t .destinations) ((plane m t) .destinations).numel ∧
  (s≠t → SpanSeparate (base m s .output) ((plane m s) .output).numel
    (base m t .output) ((plane m t) .output).numel)

structure InputContract (m : Inputs) (w : Nat → Word) : Prop where
  common : ∀ s, DestindexCopyKv1.InputContract (plane m s) w
  separated : Separated m

inductive InputError where | tensors | nope | rope | overlap
open Classical in
def checkInputs (m : Inputs) (w : Nat → Word) : Except InputError Unit :=
  if ¬DestindexCopyKv1.InputContract (plane m .nope) w then .error .nope
  else if ¬DestindexCopyKv1.InputContract (plane m .rope) w then .error .rope
  else if ¬Separated m then .error .overlap
  else .ok ()

def checkArguments (m : Option Inputs) (w : Nat → Word) : Except InputError Unit :=
  match m with | none => .error .tensors | some m => checkInputs m w

def Admitted (m : Option Inputs) (w : Nat → Word) : Prop :=
  match m with | none => False | some m => InputContract m w

structure TensorBinding (m : Inputs) (w : Nat → Word) (before : Memory) : Prop where
  planeBinding : ∀ s, DestindexCopyKv1.TensorBinding (plane m s) w before
end
end VeriTile.Bench.Repairs.DestindexCopy
