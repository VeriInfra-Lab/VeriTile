import bench.repairs.tritonbench_g.kv_cache_copy.Kernels
import VeriTile.Triton.Float.ProgramIO
import Mathlib.Tactic.FinCases
import VeriTile.Triton.Float.LogicalProgramIO

/-! Public IO of the original complete K/V split kernel. Metadata stays signed
INT32 in the source; payload loads/stores stay FP32. Real is the copy's
mathematical interpretation. Addresses below are pure logical formulas, and
all split lanes are present in both input and output channels. -/
namespace VeriTile.Bench.Repairs.KvCacheCopy.CopyIO
open VeriTile.Triton FP.Structural Source
noncomputable section

def ids (seqPid headPid : Nat) (ax : Nat) : Nat :=
  if ax = 0 then seqPid else if ax = 1 then headPid else 0

def tableAddress (p : Parameters) (seqPid : Nat) (length : Int) : Nat :=
  seqPid*p.strideBts+((length-1).tdiv p.blockSize*p.strideBtb).toNat -
    (-((length-1).tdiv p.blockSize*p.strideBtb)).toNat

def inputAddress (p : Parameters) (seqPid headPid : Nat) (key : Bool)
    (q : Nat) (i : TileIndex [p.width]) : Nat :=
  if key then seqPid*p.strideKt+headPid*p.strideKh+(q*p.width+i.1.val)*p.strideKd
  else seqPid*p.strideVt+headPid*p.strideVh+(q*p.width+i.1.val)*p.strideVd

def outputAddress (p : Parameters) (headPid : Nat) (length block : Int) (key : Bool)
    (q : Nat) (i : TileIndex [p.width]) : Nat :=
  (if key then block*p.strideKcb+(headPid:Int)*p.strideKch+
    (q:Int)*p.strideKcsplit+(length-1).tmod p.blockSize*p.strideKcs+i.1.val
  else block*p.strideVcb+(headPid:Int)*p.strideVch+
    (length-1).tmod p.blockSize*p.strideVcs+(q*p.width+i.1.val)*p.strideVcd).toNat

/-- The pre-existing target injectivity premise, expressed entirely in logical
metadata and geometry. It covers the complete split range, not just one tile. -/
def OutputUnique (p : Parameters) (headPid : Nat) (length block : Int) : Prop :=
  ∀ key u v, u < p.dims/p.width → v < p.dims/p.width → ∀ i j : TileIndex [p.width],
    outputAddress p headPid length block key u i = outputAddress p headPid length block key v j →
      u = v ∧ i = j

def realBackend (rest : Algebra ℝ) : Algebra ℝ := { rest with fp32Load := id }

def io (rest : Algebra ℝ) (p : Parameters) (seqPid headPid : Nat) : ProgramIO ℝ ℝ where
  inputs := [⟨.int, [], "lengths"⟩, ⟨.int, [], "table"⟩,
    ⟨.real, [p.dims/p.width, p.width], "key"⟩,
    ⟨.real, [p.dims/p.width, p.width], "value"⟩]
  outputs := [⟨.real, [p.dims/p.width, p.width], "keyCache"⟩,
    ⟨.real, [p.dims/p.width, p.width], "valueCache"⟩]
  buffers := ["lengths", "table", "key", "value", "keyCache", "valueCache"]
  read := by
    intro port vals
    refine Fin.cases (fun _ => seqPid) (fun p1 => ?_) port
    refine Fin.cases (fun _ => tableAddress p seqPid (vals (0 : Fin 4) ())) (fun p2 => ?_) p1
    refine Fin.cases (fun i => inputAddress p seqPid headPid Bool.true i.1.val i.2) (fun p3 => ?_) p2
    have eq : p3 = (0 : Fin 1) := Fin.eq_zero p3
    subst p3
    exact fun i => inputAddress p seqPid headPid Bool.false i.1.val i.2
  inputMask := fun _ _ _ => True
  write := by
    intro port vals
    refine Fin.cases
      (fun i => outputAddress p headPid (vals (0 : Fin 4) ()) (vals (1 : Fin 4) ()) Bool.true i.1.val i.2)
      (fun p1 => ?_) port
    have eq : p1 = (0 : Fin 1) := Fin.eq_zero p1
    subst p1
    exact fun i => outputAddress p headPid (vals (0 : Fin 4) ()) (vals (1 : Fin 4) ()) Bool.false i.1.val i.2
  outputMask := fun _ _ _ => True
  scratchWrite := by intro port; exact Fin.elim0 port
  scratchMask := by intro port; exact Fin.elim0 port
  schedule := fun _ => [{
    kernel := originalKernel p
    backend := realBackend rest
    pids := ids seqPid headPid }]
  encode := id
  observe := fun _ _ => id

def domain (rest : Algebra ℝ) (p : Parameters) (seqPid headPid : Nat)
    (vals : (io rest p seqPid headPid).Inputs) : Prop :=
  OutputUnique p headPid (vals (0 : Fin 4) ()) (vals (1 : Fin 4) ())

/-- Both K and V are copies: the independent value formula is identity on each
complete split/channel input. All addressing is explicit in the public IO. -/
def copyValues {shape : TileShape} (xs : TileIndex shape → ℝ) : TileIndex shape → ℝ := xs

def reference (rest : Algebra ℝ) (p : Parameters) (seqPid headPid : Nat)
    (vals : (io rest p seqPid headPid).Inputs) : (io rest p seqPid headPid).Outputs := by
  intro port
  refine Fin.cases (copyValues (vals (2 : Fin 4))) (fun p1 => ?_) port
  have eq : p1 = (0 : Fin 1) := Fin.eq_zero p1
  subst p1
  exact copyValues (vals (3 : Fin 4))

end
end VeriTile.Bench.Repairs.KvCacheCopy.CopyIO

namespace VeriTile.Bench.Repairs.KvCacheCopy.LogicalIO
open VeriTile.Triton FP.Structural
noncomputable section

structure Inputs (p : Source.Parameters) where
  length : Int
  block : Int
  key : Fin (p.dims/p.width) → Fin p.width → ℝ
  value : Fin (p.dims/p.width) → Fin p.width → ℝ

def inputs (rest : Algebra ℝ) (p : Source.Parameters) (seqPid headPid : Nat) : (CopyIO.io rest p seqPid headPid).Inputs ≃ Inputs p where
  toFun := fun v => ⟨v (0 : Fin 4) (), v (1 : Fin 4) (), fun i0 i1 => v (2 : Fin 4) (i0,i1,()), fun i0 i1 => v (3 : Fin 4) (i0,i1,())⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.length 
    | ⟨1,_⟩ => fun j => v.block 
    | ⟨2,_⟩ => fun j => v.key j.1 j.2.1
    | ⟨3,_⟩ => fun j => v.value j.1 j.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

structure Outputs (p : Source.Parameters) where
  key : Fin (p.dims/p.width) → Fin p.width → ℝ
  value : Fin (p.dims/p.width) → Fin p.width → ℝ

def outputs (rest : Algebra ℝ) (p : Source.Parameters) (seqPid headPid : Nat) : (CopyIO.io rest p seqPid headPid).Outputs ≃ Outputs p where
  toFun := fun v => ⟨fun i0 i1 => v (0 : Fin 2) (i0,i1,()), fun i0 i1 => v (1 : Fin 2) (i0,i1,())⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.key j.1 j.2.1
    | ⟨1,_⟩ => fun j => v.value j.1 j.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

def io (rest : Algebra ℝ) (p : Source.Parameters) (seqPid headPid : Nat) : LogicalProgramIO ℝ ℝ (Inputs p) (Outputs p) where
  core := CopyIO.io rest p seqPid headPid
  inputs := inputs rest p seqPid headPid
  outputs := outputs rest p seqPid headPid

end
end VeriTile.Bench.Repairs.KvCacheCopy.LogicalIO
