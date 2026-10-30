import bench.pretrain_kernels.tasks.local_attention_baddbmm_dv.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_attention_baddbmm_dv
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  input0 : Fin 2 → Fin 16 → Fin 128 → Fin 128 → ℝ
  input1 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ
  input2 : Fin 2 → Fin 128 → Fin 16 → Fin 128 → ℝ

abbrev OutputIndex := Fin 2 × Fin 8 × Fin 128 × Fin 128

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

def input0Equiv : (Fin (2*16*128*128) → ℝ) ≃ (Fin 2 → Fin 16 → Fin 128 → Fin 128 → ℝ) :=
  ((matrixEquiv 2 (16*128*128) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 16 (128*128) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 128 (128) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 128 → ℝ))))))))

def input1Equiv : (Fin (2*8*128*128) → ℝ) ≃ (Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ) :=
  ((matrixEquiv 2 (8*128*128) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 8 (128*128) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 128 (128) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 128 → ℝ))))))))

def input2Equiv : (Fin (2*128*16*128) → ℝ) ≃ (Fin 2 → Fin 128 → Fin 16 → Fin 128 → ℝ) :=
  ((matrixEquiv 2 (128*16*128) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 128 (16*128) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 16 (128) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 128 → ℝ))))))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 3
  nOut := 1
  nScr := 18
  bufs := ["input0", "input1", "input2", "out", "vtriu", "vmasked_fill", "vbmm", "vmul", "vadd", "vto_copy", "vsoftmax", "vto_copy_1", "vbmm_2", "vsum_1", "candidate_vbaddbmm", "candidate_vto_copy", "candidate_vtriu", "candidate_vmasked_fill", "candidate_vsoftmax", "candidate_vto_copy_1", "candidate_vbmm_1", "candidate_vsum_1"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0)))
  iarity := Fin.cases (2*16*128*128) (Fin.cases (2*8*128*128) (Fin.cases (2*128*16*128) (Fin.elim0)))
  ibuf := Fin.cases ("input0") (Fin.cases ("input1") (Fin.cases ("input2") (Fin.elim0)))
  oty := fun _ => .float
  oarity := fun _ => 2*8*128*128
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (128*128) (Fin.cases (2*1*128*128) (Fin.cases (32*128*128) (Fin.cases (2*16*128*128) (Fin.cases (2*16*128*128) (Fin.cases (2*16*128*128) (Fin.cases (2*16*128*128) (Fin.cases (2*16*128*128) (Fin.cases (32*128*128) (Fin.cases (2*8*1*128*128) (Fin.cases (32*128*128) (Fin.cases (2*16*128*128) (Fin.cases (128*128) (Fin.cases (2*16*128*128) (Fin.cases (2*16*128*128) (Fin.cases (2*16*128*128) (Fin.cases (32*128*128) (Fin.cases (2*8*1*128*128) (Fin.elim0))))))))))))))))))
  sbuf := Fin.cases ("vtriu") (Fin.cases ("vmasked_fill") (Fin.cases ("vbmm") (Fin.cases ("vmul") (Fin.cases ("vadd") (Fin.cases ("vto_copy") (Fin.cases ("vsoftmax") (Fin.cases ("vto_copy_1") (Fin.cases ("vbmm_2") (Fin.cases ("vsum_1") (Fin.cases ("candidate_vbaddbmm") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vtriu") (Fin.cases ("candidate_vmasked_fill") (Fin.cases ("candidate_vsoftmax") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vbmm_1") (Fin.cases ("candidate_vsum_1") (Fin.elim0))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨input0Equiv (v (0 : Fin 3)), input1Equiv (v (1 : Fin 3)), input2Equiv (v (2 : Fin 3))⟩
  invFun := fun v => Fin.cases (input0Equiv.symm v.input0) (Fin.cases (input1Equiv.symm v.input1) (Fin.cases (input2Equiv.symm v.input2) ((fun i => Fin.elim0 i))))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨input0Equiv (input0Equiv.symm v.input0), input1Equiv (input1Equiv.symm v.input1), input2Equiv (input2Equiv.symm v.input2)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (2*8*128*128) ≃ OutputIndex :=
  (finProdFinEquiv (m := 2) (n := 8*128*128)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 8) (n := 128*128)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 128) (n := 128)).symm))))

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
  Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0)))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.bool) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0))))))))))))))))))

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
end VeriTile.Bench.PretrainKernels.local_attention_baddbmm_dv
