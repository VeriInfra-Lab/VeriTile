import bench.bug_detection.tasks.moe__scale3.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.BugDetection.moe__scale3
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  x : Fin 16 → Fin 2048 → ℝ
  router : Fin 128 → Fin 2048 → ℝ
  gate : Fin 128 → Fin 640 → Fin 2048 → ℝ
  up : Fin 128 → Fin 640 → Fin 2048 → ℝ
  down : Fin 128 → Fin 2048 → Fin 640 → ℝ
  shared_gate : Fin 512 → Fin 2048 → ℝ
  shared_up : Fin 512 → Fin 2048 → ℝ
  shared_down : Fin 2048 → Fin 512 → ℝ

abbrev OutputIndex := Fin 16 × Fin 2048

/-- Row-major indexing, with both inverses; no backing cells are dropped. -/
def matrixEquiv (m n : Nat) (α : Type) :
    (Fin (m*n) → α) ≃ (Fin m → Fin n → α) where
  toFun := fun f i j => f (finProdFinEquiv (i,j))
  invFun := fun f k => f (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2
  left_inv := by
    intro f
    funext k
    exact congrArg f (finProdFinEquiv.apply_symm_apply k)
  right_inv := by intro f; funext i j; simp

def xEquiv : (Fin (16*2048) → ℝ) ≃ (Fin 16 → Fin 2048 → ℝ) :=
  ((matrixEquiv 16 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))

def routerEquiv : (Fin (128*2048) → ℝ) ≃ (Fin 128 → Fin 2048 → ℝ) :=
  ((matrixEquiv 128 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))

def gateEquiv : (Fin (128*640*2048) → ℝ) ≃ (Fin 128 → Fin 640 → Fin 2048 → ℝ) :=
  ((matrixEquiv 128 (640*2048) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 640 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))))

def upEquiv : (Fin (128*640*2048) → ℝ) ≃ (Fin 128 → Fin 640 → Fin 2048 → ℝ) :=
  ((matrixEquiv 128 (640*2048) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 640 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))))

def downEquiv : (Fin (128*2048*640) → ℝ) ≃ (Fin 128 → Fin 2048 → Fin 640 → ℝ) :=
  ((matrixEquiv 128 (2048*640) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 2048 (640) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 640 → ℝ))))))

def shared_gateEquiv : (Fin (512*2048) → ℝ) ≃ (Fin 512 → Fin 2048 → ℝ) :=
  ((matrixEquiv 512 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))

def shared_upEquiv : (Fin (512*2048) → ℝ) ≃ (Fin 512 → Fin 2048 → ℝ) :=
  ((matrixEquiv 512 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))

def shared_downEquiv : (Fin (2048*512) → ℝ) ≃ (Fin 2048 → Fin 512 → ℝ) :=
  ((matrixEquiv 2048 (512) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 512 → ℝ))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 8
  nOut := 1
  nScr := 5
  bufs := ["x", "router", "gate", "up", "down", "shared_gate", "shared_up", "shared_down", "out", "route_index", "route_score", "route_prob", "packed_token", "packed_weight"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0))))))))
  iarity := Fin.cases (16*2048) (Fin.cases (128*2048) (Fin.cases (128*640*2048) (Fin.cases (128*640*2048) (Fin.cases (128*2048*640) (Fin.cases (512*2048) (Fin.cases (512*2048) (Fin.cases (2048*512) (Fin.elim0))))))))
  ibuf := Fin.cases ("x") (Fin.cases ("router") (Fin.cases ("gate") (Fin.cases ("up") (Fin.cases ("down") (Fin.cases ("shared_gate") (Fin.cases ("shared_up") (Fin.cases ("shared_down") (Fin.elim0))))))))
  oty := fun _ => .float
  oarity := fun _ => 16*2048
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (16*6) (Fin.cases (16*6) (Fin.cases (16*6) (Fin.cases (16) (Fin.cases (16) (Fin.elim0)))))
  sbuf := Fin.cases ("route_index") (Fin.cases ("route_score") (Fin.cases ("route_prob") (Fin.cases ("packed_token") (Fin.cases ("packed_weight") (Fin.elim0)))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨xEquiv (v (0 : Fin 8)), routerEquiv (v (1 : Fin 8)), gateEquiv (v (2 : Fin 8)), upEquiv (v (3 : Fin 8)), downEquiv (v (4 : Fin 8)), shared_gateEquiv (v (5 : Fin 8)), shared_upEquiv (v (6 : Fin 8)), shared_downEquiv (v (7 : Fin 8))⟩
  invFun := fun v => Fin.cases (xEquiv.symm v.x) (Fin.cases (routerEquiv.symm v.router) (Fin.cases (gateEquiv.symm v.gate) (Fin.cases (upEquiv.symm v.up) (Fin.cases (downEquiv.symm v.down) (Fin.cases (shared_gateEquiv.symm v.shared_gate) (Fin.cases (shared_upEquiv.symm v.shared_up) (Fin.cases (shared_downEquiv.symm v.shared_down) ((fun i => Fin.elim0 i)))))))))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨xEquiv (xEquiv.symm v.x), routerEquiv (routerEquiv.symm v.router), gateEquiv (gateEquiv.symm v.gate), upEquiv (upEquiv.symm v.up), downEquiv (downEquiv.symm v.down), shared_gateEquiv (shared_gateEquiv.symm v.shared_gate), shared_upEquiv (shared_upEquiv.symm v.shared_up), shared_downEquiv (shared_downEquiv.symm v.shared_down)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (16*2048) ≃ OutputIndex :=
  (finProdFinEquiv (m := 16) (n := 2048)).symm

def indices : ((o : Fin core.nOut) × Fin (core.oarity o)) ≃ OutputIndex where
  toFun := fun p => outputCoords p.2
  invFun := fun p => ⟨(0 : Fin 1), outputCoords.symm p⟩
  left_inv := by
    intro ⟨o,j⟩
    have ho : o = (0 : Fin 1) := Fin.eq_zero o
    subst o
    simp
  right_inv := by intro j; simp

/-- The source computes the complete declared problem in one launch. -/
def realIO : LogicalScheduledKernelIO Inputs OutputIndex where
  core := core
  schedule := fun _ => [{ pids := fun _ => 0 }]
  inputs := inputs
  indices := indices
  realOutput := fun _ => rfl

namespace FP
open VeriTile.Triton.FP.Structural

/-- fp32 loads use the project's real carrier with an explicit fp32 tag;
narrow storage and discrete inputs retain their own cell tags. -/
def inputStorage : Fin core.nIn → TileDType :=
  Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0))))))))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.nat) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.nat) (Fin.cases (.real) (Fin.elim0)))))

/-- The same complete channel layout; dtype casts remain in the source. -/
def io (α : Type) : ProgramIO α α where
  inputs := List.ofFn fun i : Fin core.nIn => ⟨inputStorage i, [core.iarity i], core.ibuf i⟩
  outputs := List.ofFn fun o : Fin core.nOut => ⟨.bf16, [core.oarity o], core.obuf o⟩
  scratch := List.ofFn fun k : Fin core.nScr => ⟨scratchStorage k, [core.sarity k], core.sbuf k⟩
  buffers := core.bufs
  read := fun _ _ j => TileShape.linearIndex _ j
  inputMask := fun _ _ _ => True
  write := fun _ _ j => TileShape.linearIndex _ j
  outputMask := fun _ _ _ => True
  scratchWrite := fun _ _ j => TileShape.linearIndex _ j
  scratchMask := fun _ _ _ => True
  schedule := fun _ => []
  encode := id
  observe := fun _ _ => id

/-- Required discrete input conditions, independent of numerical rewrite guards. -/
def admissible {α : Type} (_vals : (io α).Inputs) : Prop :=
  True

end FP
end IO
end
end VeriTile.Bench.BugDetection.moe__scale3
