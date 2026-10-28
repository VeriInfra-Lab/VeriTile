import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Source
import VeriTile.Triton.Memory.AlgorithmProgramIO
import bench.tritonbench_g.layer_norm_fwd.Proofs.RealSurface
import VeriTile.Triton.Memory.LogicalIO
import Mathlib.Logic.Equiv.Fin.Basic

/-! Finite logical arrays for all branches of the original layer-normalization
source. The ordinary grid retains independent row execution and disjoint merge.
Input aliases and inactive output strides retain their original domains. -/
namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.RealInterface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 3000000

/-- Logical matrices use row-major indexing; windows supply source strides. -/
def inputArity (M N : Nat) : Fin 9 → Nat
  | 0 | 1 | 2 => M*N
  | 3 | 4 | 5 | 6 => N
  | 7 | 8 => 2*M

def inputType : Fin 9 → ChanTy
  | 8 => .int
  | _ => .float

def inputRegion (r : Regions) : Fin 9 → RegionName
  | 0 => r.x | 1 => r.x1 | 2 => r.residual
  | 3 => r.weight | 4 => r.bias | 5 => r.weight1 | 6 => r.bias1
  | 7 => r.rowscale | 8 => r.seeds

def inputActive (f : Flags) (M : Nat) (i : Fin 9) (j : Nat) : Prop :=
  match i with
  | 0 | 3 => True
  | 1 => f.x1=Bool.true
  | 2 => f.residual=Bool.true
  | 4 => f.bias=Bool.true
  | 5 => f.w1=Bool.true
  | 6 => f.w1=Bool.true ∧ f.b1=Bool.true
  | 7 => f.rowscale=Bool.true ∧ (j<M ∨ f.x1=Bool.true)
  | 8 => f.dropout=Bool.true ∧ (j<M ∨ f.x1=Bool.true)

def inputAddress (stride : Strides) (N : Nat) (i : Fin 9) (j : Nat) : Nat :=
  match i with
  | 0 => j/N*stride.x+j%N
  | 1 => j/N*stride.x1+j%N
  | 2 => j/N*stride.residual+j%N
  | _ => j

def outputArity (M N : Nat) : Fin 7 → Nat
  | 2 | 3 => M
  | _ => M*N

def outputType : Fin 7 → ChanTy
  | 5 | 6 => .bool
  | _ => .float

def outputRegion (r : Regions) : Fin 7 → RegionName
  | 0 => r.y | 1 => r.y1 | 2 => r.mean | 3 => r.rstd
  | 4 => r.residualOut | 5 | 6 => r.dropoutMask

def outputActive (f : Flags) : Fin 7 → Prop
  | 0 | 3 => True
  | 1 => f.w1=Bool.true
  | 2 => f.rms=Bool.false
  | 4 => f.residualOut=Bool.true
  | 5 => f.dropout=Bool.true ∧ f.dropoutMask=Bool.true
  | 6 => f.dropout=Bool.true ∧ f.dropoutMask=Bool.true ∧ f.x1=Bool.true

def outputAddress (stride : Strides) (M N : Nat) (o : Fin 7) (j : Nat) : Nat :=
  match o with
  | 0 => j/N*stride.y+j%N
  | 1 => j/N*stride.y1+j%N
  | 2 | 3 => j
  | 4 => j/N*stride.residualOut+j%N
  | 5 => j
  | 6 => M*N+j

/-- Historical algorithm view retained for its existing proof helpers. The current
pasted-source Real interface is the mathematical instance of NativeInterface.io.
Seven writable channels; unchanged input/residual and seed views follow
from the exact exterior frame and are retained in the raw result helpers. -/
def io (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ) :
    AlgorithmProgramIO where
  nIn := 9
  nOut := 7
  nScr := 0
  bufs := [r.x,r.x1,r.residual,r.weight,r.bias,r.weight1,r.bias1,r.rowscale,r.seeds,
    r.y,r.y1,r.mean,r.rstd,r.residualOut,r.dropoutMask]
  ity := inputType
  iarity := inputArity M N
  ibuf := inputRegion r
  oty := outputType
  oarity := outputArity M N
  obuf := outputRegion r
  obuf_mem := by intro o; fin_cases o <;> simp [outputRegion]
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun i _ _ _ _ j => inputAddress stride N i j.val
  imask := fun i _ _ _ _ j => inputActive f M i j.val
  owin := fun o _ _ _ _ j => outputAddress stride M N o j.val
  omask := fun o _ _ _ _ _ => outputActive f o
  swin := fun i => Fin.elim0 i
  smask := fun i => Fin.elim0 i
  stages := fun _ => [.ordinary ⟨(repaired f r stride M N eps p (tile N) (tile_arange N)).eraseDType,by
    exact RealSurface.erase_projection _ (original_toAlgorithm_supported f r stride M N eps p (tile N) (tile_arange N))⟩ (WholeGrid.grid M)]

abbrev Arrays (M N : Nat) := (i : Fin 9) → Fin (inputArity M N i) → (inputType i).carrier

def lookup {α : Type} [Zero α] {n : Nat} (a : Fin n → α) (i : Nat) : α :=
  if h : i<n then a ⟨i,h⟩ else 0

/-- Independent tensor formula: finite logical inputs supply the existing
column sums, canonical Philox, normalization and affine functions. -/
def logicalInputs (M N : Nat) (a : Arrays M N) : Inputs where
  x := fun i j => lookup (a 0) (i*N+j)
  x1 := fun i j => lookup (a 1) (i*N+j)
  residual := fun i j => lookup (a 2) (i*N+j)
  weight := lookup (a 3)
  bias := lookup (a 4)
  weight1 := lookup (a 5)
  bias1 := lookup (a 6)
  rowscale := lookup (a 7)
  seeds := lookup (a 8)

def formula (f : Flags) (M N : Nat) (eps p : ℝ) (a : Arrays M N) :
    (o : Fin 7) → Fin (outputArity M N o) → (outputType o).carrier
  | 0,j => Formula.y f (logicalInputs M N a) M N eps p (j.val/N) (j.val%N)
  | 1,j => Formula.y1 f (logicalInputs M N a) M N eps p (j.val/N) (j.val%N)
  | 2,j => Formula.mean f (logicalInputs M N a) M N p j.val
  | 3,j => Formula.rstd f (logicalInputs M N a) M N eps p j.val
  | 4,j => Formula.combined f (logicalInputs M N a) M p (j.val/N) (j.val%N)
  | 5,j => Formula.keep ((logicalInputs M N a).seeds (j.val/N)) p (j.val%N)
  | 6,j => Formula.keep ((logicalInputs M N a).seeds (M+j.val/N)) p (j.val%N)

end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.RealInterface

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.LogicalIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
open RealInterface
noncomputable section

structure Inputs (M N : Nat) where
  x : Fin M → Fin N → ℝ
  x1 : Fin M → Fin N → ℝ
  residual : Fin M → Fin N → ℝ
  weight : Fin N → ℝ
  bias : Fin N → ℝ
  weight1 : Fin N → ℝ
  bias1 : Fin N → ℝ
  rowscale : Fin (2*M) → ℝ
  seeds : Fin (2*M) → Int

def inputs (M N : Nat) : Arrays M N ≃ Inputs M N where
  toFun := fun a => ⟨fun row col => a (0:Fin 9) (finProdFinEquiv (row,col)),
    fun row col => a (1:Fin 9) (finProdFinEquiv (row,col)),
    fun row col => a (2:Fin 9) (finProdFinEquiv (row,col)),
    a (3:Fin 9),
    a (4:Fin 9),
    a (5:Fin 9),
    a (6:Fin 9),
    a (7:Fin 9),
    a (8:Fin 9)⟩
  invFun := fun a i => match i with
    | 0 => fun j => a.x (finProdFinEquiv.symm j).1 (finProdFinEquiv.symm j).2
    | 1 => fun j => a.x1 (finProdFinEquiv.symm j).1 (finProdFinEquiv.symm j).2
    | 2 => fun j => a.residual (finProdFinEquiv.symm j).1 (finProdFinEquiv.symm j).2
    | 3 => a.weight
    | 4 => a.bias
    | 5 => a.weight1
    | 6 => a.bias1
    | 7 => a.rowscale
    | 8 => a.seeds
  left_inv := by
    intro a
    funext i j
    fin_cases i <;> simp only [Prod.mk.eta, Equiv.apply_symm_apply] <;> rfl
  right_inv := by
    intro a
    cases a
    simp only [Equiv.symm_apply_apply]

structure Outputs (M N : Nat) where
  y : Fin M → Fin N → ℝ
  y1 : Fin M → Fin N → ℝ
  mean : Fin M → ℝ
  inverse : Fin M → ℝ
  residual : Fin M → Fin N → ℝ
  keep : Fin M → Fin N → Bool
  keep1 : Fin M → Fin N → Bool

def outputs (M N : Nat) : ((o : Fin 7) → Fin (outputArity M N o) → (outputType o).carrier) ≃ Outputs M N where
  toFun := fun a => ⟨fun row col => a (0:Fin 7) (finProdFinEquiv (row,col)),
    fun row col => a (1:Fin 7) (finProdFinEquiv (row,col)),
    a (2:Fin 7),
    a (3:Fin 7),
    fun row col => a (4:Fin 7) (finProdFinEquiv (row,col)),
    fun row col => a (5:Fin 7) (finProdFinEquiv (row,col)),
    fun row col => a (6:Fin 7) (finProdFinEquiv (row,col))⟩
  invFun := fun a i => match i with
    | 0 => fun j => a.y (finProdFinEquiv.symm j).1 (finProdFinEquiv.symm j).2
    | 1 => fun j => a.y1 (finProdFinEquiv.symm j).1 (finProdFinEquiv.symm j).2
    | 2 => a.mean
    | 3 => a.inverse
    | 4 => fun j => a.residual (finProdFinEquiv.symm j).1 (finProdFinEquiv.symm j).2
    | 5 => fun j => a.keep (finProdFinEquiv.symm j).1 (finProdFinEquiv.symm j).2
    | 6 => fun j => a.keep1 (finProdFinEquiv.symm j).1 (finProdFinEquiv.symm j).2
  left_inv := by
    intro a
    funext i j
    fin_cases i <;> simp only [Prod.mk.eta, Equiv.apply_symm_apply] <;> rfl
  right_inv := by
    intro a
    cases a
    simp only [Equiv.symm_apply_apply]

/-- The existing total mathematical input convention, transported from the
bijective finite tensor view. Flat-address details remain confined to IO. -/
def mathInputs (M N : Nat) (a : Inputs M N) : VeriTile.Bench.TritonBenchG.LayerNormFwd.Inputs :=
  RealInterface.logicalInputs M N ((inputs M N).symm a)

def io (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ) :
    VeriTile.Triton.LogicalIO (Arrays M N)
      ((o : Fin 7) → Fin (outputArity M N o) → (outputType o).carrier)
      (Inputs M N) (Outputs M N) (RealInterface.io f r stride M N eps p).Implements where
  inputs := inputs M N
  outputs := outputs M N
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.LogicalIO
