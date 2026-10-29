import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.StorageNumerics

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section

/-- Total local instructions on a carrier that includes all requested storage
encodings. No finite-input or exact-Real premise is used. -/
structure Primitives (α : Type) where
  fromHostBinary64 : UInt64 → α
  constant32 : ℝ → α
  addF32 : α → α → α
  subF32 : α → α → α
  mulF32 : α → α → α
  divFullF32 : α → α → α
  fmaRnF32 : α → α → α → α
  sqrtApproxFtzF32 : α → α
  greaterF32 : α → α → Bool
  philoxUniform32 : FP.Value .fp32 → α
  storeCasts : StorageCompiler.StoreCasts α

def Primitives.ops (p : Primitives α) : NativeRow.Ops α where
  literal := p.constant32
  add := p.addF32
  sub := p.subF32
  mul := p.mulF32
  div := p.divFullF32
  fma := p.fmaRnF32
  sqrt := p.sqrtApproxFtzF32
  greater := p.greaterF32
  randomValue := p.philoxUniform32
  cast32 := id
  decode := fun _ => 0

/-- Actual binary64 host words are retained before the local scalar ABI cast. -/
def Primitives.scalar (p : Primitives α) (value : Float) : α := p.fromHostBinary64 value.toBits

theorem host_word (p : Primitives α) (value : Float) : p.scalar value=p.fromHostBinary64 value.toBits := rfl

namespace Compiler
open VeriTile.Bench.Repairs.LayerNormFwd.Compiler

def lower : Term → Code
  | .input v => .read v | .literal v => .constant32 v | .random second col => .philox7 second col
  | .cast32 x => .identity32 (lower x)
  | .add x y => .addF32 (lower x) (lower y) | .sub x y => .subF32 (lower x) (lower y)
  | .mul x y => .mulF32 (lower x) (lower y) | .div x y => .divFullF32 (lower x) (lower y)
  | .fma x y z => .fmaRnF32 (lower x) (lower y) (lower z)
  | .sqrt x => .sqrtApproxFtzF32 (lower x)
  | .selectGT x y yes no => .selectGreaterF32 (lower x) (lower y) (lower yes) (lower no)
  | .at site x => .origin site (lower x)

theorem lower_bound (term : Term) : Binds term (lower term) := by
  induction term <;> constructor <;> assumption

/-- The local instruction interface has a constructor for every legal tree
and warp choice. This is a model witness, not installed-compiler conformance. -/
def canonical (schedule : NativeMachine.Schedule block) (f : VeriTile.Bench.TritonBenchG.LayerNormFwd.Flags)
    (N warps : Nat) (supported : warps∈[1,2,4,8,16,32]) : Artifact f N block where
  warps := warps
  supportedWarps := supported
  schedule := schedule
  code := fun root => lower (recipe schedule f N root)
  operands := fun root _ => lower_bound _
  stores := sourceStore f N (fun root => lower (recipe schedule f N root))
  storeOperands := fun _ => rfl
end Compiler

#axiomsClean host_word
#axiomsClean Compiler.lower_bound
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
