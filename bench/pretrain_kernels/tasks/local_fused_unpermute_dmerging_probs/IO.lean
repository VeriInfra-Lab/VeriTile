import bench.pretrain_kernels.tasks.local_fused_unpermute_dmerging_probs.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_fused_unpermute_dmerging_probs
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  parameter0 : Fin 256 → Fin 128 → ℝ
  parameter1 : Fin 256 → Fin 2048 → ℝ
  input0 : Fin 256 → Fin 2048 → ℝ

abbrev OutputIndex := Fin 256 × Fin 128

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

def parameter0Equiv : (Fin (256*128) → ℝ) ≃ (Fin 256 → Fin 128 → ℝ) :=
  ((matrixEquiv 256 (128) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 128 → ℝ))))

def parameter1Equiv : (Fin (256*2048) → ℝ) ≃ (Fin 256 → Fin 2048 → ℝ) :=
  ((matrixEquiv 256 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))

def input0Equiv : (Fin (256*2048) → ℝ) ≃ (Fin 256 → Fin 2048 → ℝ) :=
  ((matrixEquiv 256 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 3
  nOut := 1
  nScr := 45
  bufs := ["parameter0", "parameter1", "input0", "out", "vtopk_value", "vtopk_index", "vscatter", "vsort_value", "vsort_index", "vremainder", "vindex_select", "vto_copy_6", "vto_copy_7", "vsort_1_value", "vsort_1_index", "vto_copy_8", "vto_copy_9", "vgather", "vmul_1", "vsum_3", "vindex_put", "vto_copy_10", "candidate_vtopk_value", "candidate_vtopk_index", "candidate_vscatter", "candidate_vto_copy_1", "candidate_vto_copy_2", "candidate_vto_copy_3", "candidate_vsum_2", "candidate_vcumsum", "candidate_vsub", "candidate_vto_copy_4", "candidate_vcumsum_1", "candidate_vsub_1", "candidate_vadd", "candidate_vwhere", "candidate_vwhere_1", "candidate_vwhere_2", "candidate_vscatter_add", "candidate_vge", "candidate_vclamp_min", "candidate_vindex", "candidate_vto_copy_5", "candidate_vwhere_3", "candidate_vto_copy_8", "candidate_vto_copy_9", "candidate_vmul_1", "candidate_vsum_4", "candidate_vto_copy_10"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0)))
  iarity := Fin.cases (256*128) (Fin.cases (256*2048) (Fin.cases (256*2048) (Fin.elim0)))
  ibuf := Fin.cases ("parameter0") (Fin.cases ("parameter1") (Fin.cases ("input0") (Fin.elim0)))
  oty := fun _ => .float
  oarity := fun _ => 256*128
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (256*6) (Fin.cases (256*6) (Fin.cases (256*128) (Fin.cases (32768) (Fin.cases (32768) (Fin.cases (1536) (Fin.cases (1536*2048) (Fin.cases (1536*2048) (Fin.cases (32768) (Fin.cases (32768) (Fin.cases (32768) (Fin.cases (256*2048) (Fin.cases (256*2048) (Fin.cases (1536*2048) (Fin.cases (1536*2048) (Fin.cases (1536*1) (Fin.cases (32768) (Fin.cases (256*128) (Fin.cases (256*6) (Fin.cases (256*6) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128*2048) (Fin.cases (1537*2048) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (32768*2048) (Fin.cases (256*128*2048) (Fin.cases (256*128*2048) (Fin.cases (256*2048) (Fin.cases (256*2048) (Fin.cases (256*128*2048) (Fin.cases (256*128*1) (Fin.cases (256*128) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vtopk_value") (Fin.cases ("vtopk_index") (Fin.cases ("vscatter") (Fin.cases ("vsort_value") (Fin.cases ("vsort_index") (Fin.cases ("vremainder") (Fin.cases ("vindex_select") (Fin.cases ("vto_copy_6") (Fin.cases ("vto_copy_7") (Fin.cases ("vsort_1_value") (Fin.cases ("vsort_1_index") (Fin.cases ("vto_copy_8") (Fin.cases ("vto_copy_9") (Fin.cases ("vgather") (Fin.cases ("vmul_1") (Fin.cases ("vsum_3") (Fin.cases ("vindex_put") (Fin.cases ("vto_copy_10") (Fin.cases ("candidate_vtopk_value") (Fin.cases ("candidate_vtopk_index") (Fin.cases ("candidate_vscatter") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vto_copy_2") (Fin.cases ("candidate_vto_copy_3") (Fin.cases ("candidate_vsum_2") (Fin.cases ("candidate_vcumsum") (Fin.cases ("candidate_vsub") (Fin.cases ("candidate_vto_copy_4") (Fin.cases ("candidate_vcumsum_1") (Fin.cases ("candidate_vsub_1") (Fin.cases ("candidate_vadd") (Fin.cases ("candidate_vwhere") (Fin.cases ("candidate_vwhere_1") (Fin.cases ("candidate_vwhere_2") (Fin.cases ("candidate_vscatter_add") (Fin.cases ("candidate_vge") (Fin.cases ("candidate_vclamp_min") (Fin.cases ("candidate_vindex") (Fin.cases ("candidate_vto_copy_5") (Fin.cases ("candidate_vwhere_3") (Fin.cases ("candidate_vto_copy_8") (Fin.cases ("candidate_vto_copy_9") (Fin.cases ("candidate_vmul_1") (Fin.cases ("candidate_vsum_4") (Fin.cases ("candidate_vto_copy_10") (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨parameter0Equiv (v (0 : Fin 3)), parameter1Equiv (v (1 : Fin 3)), input0Equiv (v (2 : Fin 3))⟩
  invFun := fun v => Fin.cases (parameter0Equiv.symm v.parameter0) (Fin.cases (parameter1Equiv.symm v.parameter1) (Fin.cases (input0Equiv.symm v.input0) ((fun i => Fin.elim0 i))))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨parameter0Equiv (parameter0Equiv.symm v.parameter0), parameter1Equiv (parameter1Equiv.symm v.parameter1), input0Equiv (input0Equiv.symm v.input0)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (256*128) ≃ OutputIndex :=
  (finProdFinEquiv (m := 256) (n := 128)).symm

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
  Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.elim0)))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))

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

/-- Capacity in the public mathematical metadata view. The decoder must be
bound to the selected numerical models; this is not an implicit conversion. -/
def metadataAdmissible {α : Type} (decode : α → ℝ) (vals : (io α).Inputs) : Prop :=
  (let parameter0 := parameter0Equiv (fun j : Fin 32768 => decode (vals (0 : Fin 3) (j, PUnit.unit)));
let vtopk_index : Fin 256 → Fin 6 → Int := fun i0 i1 => ((((List.finRange 128).mergeSort (fun left right => decide (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val : Int);
let vscatter : Fin 256 → Fin 128 → Bool := fun i0 i1 => ((List.finRange 6).foldl (fun value lane => if vtopk_index i0 ⟨(lane.val) % 6, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then Bool.true else value) (Bool.false));
let vto_copy_7 : Fin 32768 → Int := fun i0 => (if vscatter ⟨((i0.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then 1 else 0);
(∑ expert : Fin 32768, ((vto_copy_7 ⟨(expert.val) % 32768, Nat.mod_lt _ (by decide)⟩ + 0) / 1) * 1) = 1536)

end FP
end IO
end
end VeriTile.Bench.PretrainKernels.local_fused_unpermute_dmerging_probs
