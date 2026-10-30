import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Contract
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.TensorBinding
open VeriTile.Triton VeriTile.Torch
noncomputable section

variable (x y : VeriTile.Torch.FP32.Tensor) (lx : VeriTile.Torch.FP32.TensorLaws x) (ly : VeriTile.Torch.FP32.TensorLaws y)
    (h : InputContract x.metadata y.metadata)

def matrixView : VeriTile.Torch.FP32.DataView := VeriTile.Torch.FP32.dataView x lx h.contiguous.1
def vectorView : VeriTile.Torch.FP32.DataView := VeriTile.Torch.FP32.dataView y ly h.contiguous.2

def matrixBase : Nat := (matrixView x y lx h).base / 4
def vectorBase : Nat := (vectorView x y ly h).base / 4

theorem matrix_base_bytes : 4 * matrixBase x y lx h = (matrixView x y lx h).base := by
  have ha := VeriTile.Torch.FP32.dataView_aligned x lx h.contiguous.1 h.alignment.1
  change (matrixView x y lx h).base % 4 = 0 at ha
  have hd := Nat.mod_add_div (matrixView x y lx h).base 4
  simp only [ha, Nat.zero_add] at hd
  exact hd

theorem vector_base_bytes : 4 * vectorBase x y ly h = (vectorView x y ly h).base := by
  have ha := VeriTile.Torch.FP32.dataView_aligned y ly h.contiguous.2 h.alignment.2
  change (vectorView x y ly h).base % 4 = 0 at ha
  have hd := Nat.mod_add_div (vectorView x y ly h).base 4
  simp only [ha, Nat.zero_add] at hd
  exact hd

/-- Nonempty data pointers are the same observations used by admission. -/
theorem matrix_pointer (nonempty : 0 < x.metadata.numel) :
    BitVec.ofNat 64 (4 * matrixBase x y lx h) = x.dataPtr := by
  rw [matrix_base_bytes]
  simpa [VeriTile.Torch.FP32.DataView.address, matrixView] using VeriTile.Torch.FP32.actual_pointer x lx h.contiguous.1 nonempty

theorem vector_pointer (nonempty : 0 < y.metadata.numel) :
    BitVec.ofNat 64 (4 * vectorBase x y ly h) = y.dataPtr := by
  rw [vector_base_bytes]
  simpa [VeriTile.Torch.FP32.DataView.address, vectorView] using VeriTile.Torch.FP32.actual_pointer y ly h.contiguous.2 nonempty

def memory {α : Type} (decode : VeriTile.Torch.FP32.Word → α) (read : Nat → VeriTile.Torch.FP32.Word) : Nat → α :=
  fun addr => decode (read (4*addr))

def matrix {α : Type} (decode : VeriTile.Torch.FP32.Word → α) (read : Nat → VeriTile.Torch.FP32.Word) : Nat → Nat → α :=
  fun i j => decode (VeriTile.Torch.FP32.logicalValue read x lx (i * columns x.metadata + j))

def vector {α : Type} (decode : VeriTile.Torch.FP32.Word → α) (read : Nat → VeriTile.Torch.FP32.Word) : Nat → α :=
  fun j => decode (VeriTile.Torch.FP32.logicalValue read y ly j)

include h in
theorem matrix_index (i j : Nat) (hi : i < rows x.metadata)
    (hj : j < columns x.metadata) : i*columns x.metadata+j < x.metadata.numel := by
  rw [matrix_numel x.metadata y.metadata h]
  have hm := Nat.mul_le_mul_right (columns x.metadata) (Nat.succ_le_of_lt hi)
  rw [Nat.succ_mul] at hm
  omega

#axiomsClean matrix_base_bytes
#axiomsClean vector_base_bytes
#axiomsClean matrix_pointer
#axiomsClean vector_pointer
#axiomsClean matrix_index
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.TensorBinding
