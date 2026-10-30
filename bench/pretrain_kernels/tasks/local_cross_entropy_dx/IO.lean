import bench.pretrain_kernels.tasks.local_cross_entropy_dx.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_cross_entropy_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  parameter0 : Fin 128 → Fin 2 → Int
  input0 : Fin 128 → Fin 2 → Fin 1024 → ℝ
  input1 : Fin 128 → Fin 2 → ℝ

abbrev OutputIndex := Fin 128 × Fin 2 × Fin 1024

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

def parameter0Equiv : (Fin (128*2) → Int) ≃ (Fin 128 → Fin 2 → Int) :=
  ((matrixEquiv 128 (2) Int).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2 → Int))))

def input0Equiv : (Fin (128*2*1024) → ℝ) ≃ (Fin 128 → Fin 2 → Fin 1024 → ℝ) :=
  ((matrixEquiv 128 (2*1024) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 2 (1024) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 1024 → ℝ))))))

def input1Equiv : (Fin (128*2) → ℝ) ≃ (Fin 128 → Fin 2 → ℝ) :=
  ((matrixEquiv 128 (2) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2 → ℝ))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 3
  nOut := 1
  nScr := 45
  bufs := ["parameter0", "input0", "input1", "out", "vto_copy", "vmax_1_value", "vmax_1_index", "vsub", "vexp", "vsum_1", "vto_copy_1", "vneg_1", "vneg_2", "vsub_3", "vexp_1", "vdiv", "vmul", "vneg_3", "vsum_2", "vadd", "vneg_4", "vscatter_add", "vgather_1", "vadd_1", "vadd_2", "vscatter", "vgather_2", "vadd_3", "vadd_4", "vscatter_1", "vto_copy_2", "candidate_vto_copy", "candidate_vmax_1_value", "candidate_vmax_1_index", "candidate_vsub", "candidate_vexp", "candidate_vsum_1", "candidate_vto_copy_1", "candidate_vto_copy_2", "candidate_vsub_3", "candidate_vexp_1", "candidate_vdiv", "candidate_veq", "candidate_vsub_4", "candidate_vwhere_1", "candidate_vne_1", "candidate_vmul", "candidate_vwhere_2", "candidate_vto_copy_3"]
  ity := Fin.cases (.int) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0)))
  iarity := Fin.cases (128*2) (Fin.cases (128*2*1024) (Fin.cases (128*2) (Fin.elim0)))
  ibuf := Fin.cases ("parameter0") (Fin.cases ("input0") (Fin.cases ("input1") (Fin.elim0)))
  oty := fun _ => .float
  oarity := fun _ => 128*2*1024
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (128*2*1024) (Fin.cases (128*2*1) (Fin.cases (128*2*1) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2) (Fin.cases (128*2) (Fin.cases (128*2) (Fin.cases (128*2) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2*1) (Fin.cases (128*2*1) (Fin.cases (128*2*1024) (Fin.cases (128*2*1) (Fin.cases (128*2*1) (Fin.cases (128*2*1) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2*1) (Fin.cases (128*2*1) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2*1) (Fin.cases (128*2*1) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2*1) (Fin.cases (128*2) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.cases (128*2*1024) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vto_copy") (Fin.cases ("vmax_1_value") (Fin.cases ("vmax_1_index") (Fin.cases ("vsub") (Fin.cases ("vexp") (Fin.cases ("vsum_1") (Fin.cases ("vto_copy_1") (Fin.cases ("vneg_1") (Fin.cases ("vneg_2") (Fin.cases ("vsub_3") (Fin.cases ("vexp_1") (Fin.cases ("vdiv") (Fin.cases ("vmul") (Fin.cases ("vneg_3") (Fin.cases ("vsum_2") (Fin.cases ("vadd") (Fin.cases ("vneg_4") (Fin.cases ("vscatter_add") (Fin.cases ("vgather_1") (Fin.cases ("vadd_1") (Fin.cases ("vadd_2") (Fin.cases ("vscatter") (Fin.cases ("vgather_2") (Fin.cases ("vadd_3") (Fin.cases ("vadd_4") (Fin.cases ("vscatter_1") (Fin.cases ("vto_copy_2") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vmax_1_value") (Fin.cases ("candidate_vmax_1_index") (Fin.cases ("candidate_vsub") (Fin.cases ("candidate_vexp") (Fin.cases ("candidate_vsum_1") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vto_copy_2") (Fin.cases ("candidate_vsub_3") (Fin.cases ("candidate_vexp_1") (Fin.cases ("candidate_vdiv") (Fin.cases ("candidate_veq") (Fin.cases ("candidate_vsub_4") (Fin.cases ("candidate_vwhere_1") (Fin.cases ("candidate_vne_1") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vwhere_2") (Fin.cases ("candidate_vto_copy_3") (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨parameter0Equiv (v (0 : Fin 3)), input0Equiv (v (1 : Fin 3)), input1Equiv (v (2 : Fin 3))⟩
  invFun := fun v => Fin.cases (parameter0Equiv.symm v.parameter0) (Fin.cases (input0Equiv.symm v.input0) (Fin.cases (input1Equiv.symm v.input1) ((fun i => Fin.elim0 i))))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨parameter0Equiv (parameter0Equiv.symm v.parameter0), input0Equiv (input0Equiv.symm v.input0), input1Equiv (input1Equiv.symm v.input1)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (128*2*1024) ≃ OutputIndex :=
  (finProdFinEquiv (m := 128) (n := 2*1024)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 2) (n := 1024)).symm))

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
  Fin.cases (.int) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0)))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))

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
def admissible {α : Type} (vals : (io α).Inputs) : Prop :=
  (∀ j : Fin 256, Int.le 0 (vals (0 : Fin 3) (j, PUnit.unit) : Int) ∧ Int.lt (vals (0 : Fin 3) (j, PUnit.unit) : Int) 1024)

end FP
end IO
end
end VeriTile.Bench.PretrainKernels.local_cross_entropy_dx
