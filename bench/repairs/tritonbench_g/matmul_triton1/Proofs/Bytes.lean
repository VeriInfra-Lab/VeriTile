import bench.repairs.tritonbench_g.matmul_triton1.Proofs.Backing

namespace VeriTile.Bench.Repairs.MatmulTriton1.Kernel
open VeriTile.Triton FP.Structural
open Revision001 SharedContract TensorBinding
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Little-endian bytes of the same aligned physical FP32-word memory. -/
def byte (w : Word) (k : Nat) : BitVec 8 := BitVec.extractLsb' (8*k) 8 w.bits
def observeByte (read : Memory Word) (q : Nat) : BitVec 8 :=
  byte (read (q-q%4)) (q%4)

theorem output_aligned (x : Launch) (nonempty : 0 < x.c.metadata.numel) : base x.c % 4 = 0 := by
  have h := VeriTile.Torch.FP32.dataView_aligned x.c x.lc x.outputContiguous x.outputAligned
  simpa [VeriTile.Torch.FP32.DataView.base,VeriTile.Torch.FP32.dataView,base,
    show x.c.metadata.numel ≠ 0 by omega] using h

theorem byte_at (read : Memory Word) (t : Tensor) (aligned : base t % 4 = 0)
    (i : Nat) (k : Fin 4) :
    observeByte read (address t i+k.val) = byte (read (address t i)) k.val := by
  have hk := k.isLt
  have hm : (address t i+k.val)%4 = k.val := by unfold address; omega
  have hs : address t i+k.val-k.val = address t i := by omega
  simp [observeByte,hm,hs]

theorem image_empty {α : Type} [Inhabited α] (x : Launch) (read : Memory α)
    (mem : RegionName → Nat → Cell α) (empty : x.c.metadata.numel = 0) : image x read mem = read := by
  funext q
  apply image_frame
  omega

theorem byte_frame (x : Launch) (read : Memory Word)
    (mem : RegionName → Nat → Cell Word) (q : Nat)
    (outside : q < base x.c ∨ base x.c+4*x.c.metadata.numel ≤ q) :
    observeByte (image x read mem) q = observeByte read q := by
  by_cases empty : x.c.metadata.numel = 0
  · rw [image_empty x read mem empty]
  · have align := output_aligned x (by omega)
    unfold observeByte
    rw [image_frame x read mem (q-q%4) (by omega)]

theorem output_bytes (x : Launch) (read : Memory Word) (before final : State Word)
    (value : Index x → Word) (result : Post x before final value) (i : Index x) (k : Fin 4) :
    observeByte (image x read final.mem) (address x.c (offset x i)+k.val) = byte (value i) k.val := by
  have hi := offset_bound x i
  rw [byte_at _ x.c (output_aligned x (by omega)),image_at x read final.mem _ hi,
    result.1 i,Cell.read_same]

theorem input_storage_bytes (x : Launch) (read : Memory Word)
    (mem : RegionName → Nat → Cell Word) (t : Tensor) (member : t=x.a ∨ t=x.b)
    (q : Nat) (hq : storageContains t q) :
    observeByte (image x read mem) q = observeByte read q := by
  apply byte_frame
  have sep := x.storageSeparate t member q
  by_contra outside
  exact sep (by omega) (by omega) hq

#axiomsClean byte_frame
#axiomsClean output_bytes
#axiomsClean input_storage_bytes
end
end VeriTile.Bench.Repairs.MatmulTriton1.Kernel
