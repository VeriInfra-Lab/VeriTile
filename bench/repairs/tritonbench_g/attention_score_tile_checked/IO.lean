import VeriTile.Triton.Memory.LogicalIO
import bench.repairs.tritonbench_g.attention_score_tile_checked.Kernels
import VeriTile.Triton.Memory.AlgorithmProgramIO
/-! Full score-grid Real IO with the original input roles and empty dispatch.
The already allocated zero output is an explicit input only when the source
grid is skipped. No new initializer, wrapper executor or source is introduced. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.RealInterface
open VeriTile Triton Reused Geometry Physical
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000

def arity (c : Shape) : Fin 4 → Nat
  | ⟨0,_⟩ => c.querySize
  | ⟨1,_⟩ => c.keySize
  | ⟨2,_⟩ => c.normalizerSize
  | ⟨3,_⟩ => c.outputSize
def region : Fin 4 → RegionName
  | ⟨0,_⟩ => "q"
  | ⟨1,_⟩ => "k"
  | ⟨2,_⟩ => "m"
  | ⟨3,_⟩ => "out"
abbrev Inputs (α : Type) (c : Shape) := (i : Fin 4) → Fin (arity c i) → α
abbrev Outputs (α : Type) (c : Shape) := Fin 1 → Fin c.outputSize → α

def empty (c : Shape) : Prop := c.N=0 ∨ c.outputSize=0
instance (c : Shape) : Decidable (empty c) := inferInstanceAs (Decidable (_ ∨ _))

def outputCoordinates (c : Shape) (i : Fin c.outputSize) : Fin c.Z × Fin c.H × Fin c.NK :=
  let outer : Fin (c.Z*c.H) × Fin c.NK := finProdFinEquiv.symm i
  let inner : Fin c.Z × Fin c.H := finProdFinEquiv.symm outer.1
  (inner.1,inner.2,outer.2)

theorem output_address (c : Shape) (i : Fin c.outputSize) :
    c.outputBase (outputCoordinates c i).1.val (outputCoordinates c i).2.1.val +
      (outputCoordinates c i).2.2.val = i.val := by
  let outer : Fin (c.Z*c.H) × Fin c.NK := finProdFinEquiv.symm i
  let inner : Fin c.Z × Fin c.H := finProdFinEquiv.symm outer.1
  have ho := congrArg Fin.val (finProdFinEquiv.apply_symm_apply i)
  have hi := congrArg Fin.val (finProdFinEquiv.apply_symm_apply outer.1)
  change c.outputBase inner.1.val inner.2.val + outer.2.val = i.val
  change inner.2.val+c.H*inner.1.val=outer.1.val at hi
  change outer.2.val+c.NK*outer.1.val=i.val at ho
  have hi' : inner.1.val*c.H+inner.2.val=outer.1.val := by simpa [Nat.mul_comm,Nat.add_comm] using hi
  rw [Shape.outputBase,hi']
  simpa [Nat.mul_comm,Nat.add_comm] using ho

theorem key_all_heads_bound (c : Shape) (z : Fin c.Z) (h : Fin c.HK)
    (j : Fin c.NK) (d : Fin c.D) :
    (z.val*c.HK+h.val)*c.NK*c.D+j.val*c.D+d.val < c.keySize := by
  have hz := digit_bound _ _ _ _ z.isLt h.isLt
  have hj := digit_bound _ _ _ _ hz j.isLt
  have hd := digit_bound _ _ _ _ hj d.isLt
  simpa [Shape.keySize,Nat.add_mul,Nat.add_assoc] using hd

def logicalInputs {α : Type} (c : Shape) (xs : Inputs α c) : LogicalInputs α c where
  q := fun z h i d => xs 0 ⟨_,query_bound c z h i d⟩
  k := fun z h j d => xs 1 ⟨_,key_all_heads_bound c z h j d⟩
  m := fun z h i => xs 2 ⟨_,digit_bound _ _ _ _ (digit_bound _ _ _ _ z.isLt h.isLt) i.isLt⟩

def initialOutput (c : Shape) (xs : Inputs ℝ c) : Prop :=
  empty c → ∀ i, xs 3 i=0

/-- Exact Real projection of the authoritative shared quotation. -/
def source (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) : AlgorithmSource :=
  ⟨(kernel c offset size scale window complement).eraseDType,by
    rw [shared_source]
    exact RealSource.projection c.B c.D (parameters c offset size scale window complement) Bool.false⟩

def launch (c : Shape) (job : WholeGrid.Job c) : AlgorithmLaunch where
  pids := fun dim => if dim=0 then job.2.2.val else if dim=1 then job.1.val*c.H+job.2.1.val else 0

def programIO (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) :
    AlgorithmProgramIO where
  nIn := 4
  nOut := 1
  nScr := 0
  bufs := ["q","k","m","out"]
  ity := fun _ => .float
  iarity := arity c
  ibuf := region
  oty := fun _ => .float
  oarity := fun _ => c.outputSize
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := fun i => Fin.elim0 i
  sbuf := fun i => Fin.elim0 i
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun i _ _ _ _ _ => i.val≠3 ∨ empty c
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun i => Fin.elim0 i
  smask := fun i => Fin.elim0 i
  initialUndef := .preserve
  stages := fun _ => if empty c then [] else (WholeGrid.jobs c).map fun job =>
    .single (source c offset size scale window complement) (launch c job)

end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.RealInterface

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.LogicalIO
open VeriTile Triton Reused Geometry Physical
noncomputable section

/-- Normalizers use exactly N mathematical queries; the backing record retains
its larger allocated normalizer capacity and the empty-dispatch output input. -/
structure Arrays (c : Shape) where
  query : Fin c.Z → Fin c.H → Fin c.N → Fin c.D → ℝ
  key : Fin c.Z → Fin c.HK → Fin c.NK → Fin c.D → ℝ
  normalizer : Fin c.Z → Fin c.H → Fin c.N → ℝ

def arraysOfBacking (c : Shape) (raw : RealInterface.Inputs ℝ c) : Arrays c where
  query := (RealInterface.logicalInputs c raw).q
  key := (RealInterface.logicalInputs c raw).k
  normalizer := fun z h row => (RealInterface.logicalInputs c raw).m z h
    ⟨row.val,lt_of_lt_of_le row.isLt c.normalizerCapacity⟩

structure Inputs (c : Shape) where
  arrays : Arrays c
  backing : RealInterface.Inputs ℝ c
  connection : arrays = arraysOfBacking c backing

def Inputs.query (xs : Inputs c) := xs.arrays.query
def Inputs.key (xs : Inputs c) := xs.arrays.key
def Inputs.normalizer (xs : Inputs c) := xs.arrays.normalizer

def keyHead (c : Shape) (h : Fin c.H) : Fin c.HK :=
  ⟨h.val/(c.H/c.HK),grouped_head c h⟩

abbrev Outputs (c : Shape) := Fin c.Z → Fin c.H → Fin c.NK → ℝ

def inputsEquiv (c : Shape) : RealInterface.Inputs ℝ c ≃ Inputs c where
  toFun := fun raw => ⟨arraysOfBacking c raw,raw,rfl⟩
  invFun := Inputs.backing
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨arrays,backing,connection⟩
    cases connection
    rfl

def outputsEquiv (c : Shape) : RealInterface.Outputs ℝ c ≃ Outputs c where
  toFun := fun values z h key => values 0 (finProdFinEquiv (finProdFinEquiv (z,h),key))
  invFun := fun values _ i =>
    let coord := RealInterface.outputCoordinates c i
    values coord.1 coord.2.1 coord.2.2
  left_inv := by
    intro values
    funext o i
    have zero : o = 0 := Fin.eq_zero o
    subst o
    simp only [RealInterface.outputCoordinates,Prod.mk.eta,Equiv.apply_symm_apply]
  right_inv := by
    intro values
    funext z h key
    simp only [RealInterface.outputCoordinates,Equiv.symm_apply_apply]

def initialOutput (c : Shape) (xs : Inputs c) : Prop :=
  RealInterface.initialOutput c xs.backing

def io (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) :
    LogicalDomainIO (RealInterface.Inputs ℝ c) (RealInterface.Outputs ℝ c)
      (Inputs c) (Outputs c)
      (RealInterface.programIO c offset size scale window complement).ImplementsOn where
  inputs := inputsEquiv c
  outputs := outputsEquiv c

end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.LogicalIO
