import bench.repairs.tritonbench_g.rmsnorm_triton.Geometry
import bench.tritonbench_g.rmsnorm_triton.Proofs.Source
import bench.tritonbench_g.rmsnorm_triton.Proofs.WholeHost
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import Mathlib.Logic.Equiv.Fin.Basic

/-! Full-grid regional IO: one N-element channel per logical row, shared
N-element weights, all output rows, and no private scratch. Input/weight
aliases remain possible through coherent logical read observations. -/
namespace VeriTile.Bench.Repairs.RmsnormTriton.RmsIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormTriton
open Whole
open KernelGrid
noncomputable section

def rowCount (g : Geometry) : Nat := g.batches*g.rows
def batch (g : Geometry) (q : Nat) : Nat := q/g.rows
def row (g : Geometry) (q : Nat) : Nat := q%g.rows
def inputAddress (g : Geometry) (q k : Nat) : Nat :=
  batch g q*g.input.stride 0+row g q*g.input.stride 1+k*g.input.stride 2
def outputAddress (g : Geometry) (q k : Nat) : Nat :=
  batch g q*g.output.stride 0+row g q*g.output.stride 1+k*g.output.stride 2

def io (g : Geometry) : UKernelIO where
  kernel := (kernel "x" "weights" "out" (g.input.stride 0) (g.input.stride 1)
    (g.input.stride 2) (g.weights.stride 0) (g.output.stride 0) (g.output.stride 1)
    (g.output.stride 2) g.width blockSize g.eps).eraseDType
  projection := typed_supported .fp32 .fp32 .fp32 Bool.false "x" "weights" "out"
    (g.input.stride 0) (g.input.stride 1) (g.input.stride 2) (g.weights.stride 0)
    (g.output.stride 0) (g.output.stride 1) (g.output.stride 2) g.width blockSize g.eps
  nIn := rowCount g+1
  nOut := rowCount g
  nScr := 0
  bufs := ["x","weights","out"]
  ity := fun _ => .float
  iarity := fun _ => g.width
  ibuf := fun i => if i.val<rowCount g then "x" else "weights"
  oty := fun _ => .float
  oarity := fun _ => g.width
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun i _ _ _ _ j =>
    if i.val<rowCount g then inputAddress g i.val j.val else j.val*g.weights.stride 0
  imask := fun _ _ _ _ _ _ => 0<rowCount g
  owin := fun o _ _ _ _ j => outputAddress g o.val j.val
  omask := fun _ _ _ _ _ _ => True
  swin := by intro t; exact Fin.elim0 t
  smask := by intro t; exact Fin.elim0 t

def launch (b m : Nat) : AlgorithmLaunch where
  pids := fun ax => if ax=0 then b else if ax=1 then m else 0

def schedule (g : Geometry) (work : Schedule g.parameters) :
    (io g).Ctx → List AlgorithmLaunch := fun _ => work.order.map (fun bm => launch bm.1 bm.2)

/-- Independent unpadded RMS normalization; no source loop or state appears. -/
def rms {N : Nat} (eps : ℝ) (xs ws : Fin N → ℝ) (k : Fin N) : ℝ :=
  (xs k * (Real.sqrt ((∑ j : Fin N, xs j * xs j)/(N : ℝ)+eps))⁻¹) * ws k

def target (g : Geometry) : (io g).Ctx → (o : Fin (io g).nOut) →
    Fin ((io g).oarity o) → ((io g).oty o).carrier :=
  fun vals o k => rms g.eps (vals o.castSucc) (vals (Fin.last (rowCount g))) k


/-- Split all row channels from the shared weight channel. -/
def rowInputs (g : Geometry) : (io g).Ctx ≃
    ((Fin (rowCount g) → Fin g.width → ℝ) × (Fin g.width → ℝ)) := by
  change (Fin (rowCount g+1) → Fin g.width → ℝ) ≃ _
  exact {
    toFun := fun vals => (fun row => vals row.castSucc, vals (Fin.last (rowCount g)))
    invFun := fun values => Fin.lastCases values.2 values.1
    left_inv := by
      intro vals
      funext channel
      refine Fin.lastCases ?_ (fun row => ?_) channel <;> simp
    right_inv := by intro values; cases values; simp }

/-- Logical batch/row coordinates and shared weights, with a full inverse. -/
def logicalInputs (g : Geometry) : (io g).Ctx ≃
    ((Fin g.batches → Fin g.rows → Fin g.width → ℝ) × (Fin g.width → ℝ)) :=
  (rowInputs g).trans (Equiv.prodCongr
    ((Equiv.arrowCongr finProdFinEquiv.symm (Equiv.refl (Fin g.width → ℝ))).trans
      (Equiv.curry _ _ _)) (Equiv.refl _))

/-- The complete output tensor indexed by logical batch, row and feature. -/
def logicalIndices (g : Geometry) :
    ((o : Fin (io g).nOut) × Fin ((io g).oarity o)) ≃
      ((Fin g.batches × Fin g.rows) × Fin g.width) := by
  change ((o : Fin (g.batches*g.rows)) × Fin g.width) ≃ _
  exact {
  toFun := fun cell => (finProdFinEquiv.symm cell.1,cell.2)
  invFun := fun cell => ⟨finProdFinEquiv cell.1,cell.2⟩
  left_inv := by
    intro cell
    change ⟨finProdFinEquiv (finProdFinEquiv.symm cell.1),cell.2⟩ = cell
    rw [finProdFinEquiv.apply_symm_apply]
  right_inv := by
    intro cell
    change (finProdFinEquiv.symm (finProdFinEquiv cell.1),cell.2) = cell
    rw [finProdFinEquiv.symm_apply_apply] }

/-- Complete 2D schedule and unchanged regional binding, safety and frame. -/
def logicalIO (g : Geometry) (work : Schedule g.parameters) :
    LogicalScheduledKernelIO
      ((Fin g.batches → Fin g.rows → Fin g.width → ℝ) × (Fin g.width → ℝ))
      ((Fin g.batches × Fin g.rows) × Fin g.width) where
  core := io g
  schedule := schedule g work
  inputs := logicalInputs g
  indices := logicalIndices g
  realOutput := fun _ => rfl

end
end VeriTile.Bench.Repairs.RmsnormTriton.RmsIO
