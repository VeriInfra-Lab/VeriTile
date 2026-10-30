import bench.pretrain_kernels.tasks.local_pad_routing_map.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_pad_routing_map
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  input0 : Fin 128 → Fin 16 → Bool

abbrev OutputIndex := Fin 128 × Fin 16

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

def input0Equiv : (Fin (128*16) → Bool) ≃ (Fin 128 → Fin 16 → Bool) :=
  ((matrixEquiv 128 (16) Bool).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 16 → Bool))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 1
  nOut := 1
  nScr := 24
  bufs := ["input0", "out", "vto_copy", "vsum_1", "vneg", "vremainder", "veq", "vto_copy_1", "vcumsum", "vle", "vbitwise_and", "vwhere", "vto_copy_2", "candidate_vto_copy", "candidate_vsum_1", "candidate_vremainder", "candidate_vne", "candidate_vrsub", "candidate_vwhere", "candidate_veq", "candidate_vto_copy_1", "candidate_vcumsum", "candidate_vle", "candidate_vbitwise_and", "candidate_vwhere_1", "candidate_vto_copy_3"]
  ity := Fin.cases (.bool) (Fin.elim0)
  iarity := Fin.cases (128*16) (Fin.elim0)
  ibuf := Fin.cases ("input0") (Fin.elim0)
  oty := fun _ => .float
  oarity := fun _ => 128*16
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (16*128) (Fin.cases (16) (Fin.cases (16) (Fin.cases (16) (Fin.cases (16*128) (Fin.cases (16*128) (Fin.cases (16*128) (Fin.cases (16*128) (Fin.cases (16*128) (Fin.cases (16*128) (Fin.cases (128*16) (Fin.cases (16*128) (Fin.cases (16) (Fin.cases (16) (Fin.cases (16) (Fin.cases (16) (Fin.cases (16) (Fin.cases (16*128) (Fin.cases (16*128) (Fin.cases (16*128) (Fin.cases (16*128) (Fin.cases (16*128) (Fin.cases (16*128) (Fin.cases (128*16) (Fin.elim0))))))))))))))))))))))))
  sbuf := Fin.cases ("vto_copy") (Fin.cases ("vsum_1") (Fin.cases ("vneg") (Fin.cases ("vremainder") (Fin.cases ("veq") (Fin.cases ("vto_copy_1") (Fin.cases ("vcumsum") (Fin.cases ("vle") (Fin.cases ("vbitwise_and") (Fin.cases ("vwhere") (Fin.cases ("vto_copy_2") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vsum_1") (Fin.cases ("candidate_vremainder") (Fin.cases ("candidate_vne") (Fin.cases ("candidate_vrsub") (Fin.cases ("candidate_vwhere") (Fin.cases ("candidate_veq") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vcumsum") (Fin.cases ("candidate_vle") (Fin.cases ("candidate_vbitwise_and") (Fin.cases ("candidate_vwhere_1") (Fin.cases ("candidate_vto_copy_3") (Fin.elim0))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨input0Equiv (v (0 : Fin 1))⟩
  invFun := fun v => Fin.cases (input0Equiv.symm v.input0) ((fun i => Fin.elim0 i))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨input0Equiv (input0Equiv.symm v.input0)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (128*16) ≃ OutputIndex :=
  (finProdFinEquiv (m := 128) (n := 16)).symm

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
open scoped BigOperators Classical
open VeriTile.Triton.FP.Structural

/-- fp32 loads use the project's real carrier with an explicit fp32 tag;
narrow storage and discrete inputs retain their own cell tags. -/
def inputStorage : Fin core.nIn → TileDType :=
  Fin.cases (.bool) (Fin.elim0)

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.bf16) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.bf16) (Fin.elim0))))))))))))))))))))))))

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
end VeriTile.Bench.PretrainKernels.local_pad_routing_map
