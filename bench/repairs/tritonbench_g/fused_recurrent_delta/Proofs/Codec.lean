import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Physical
import VeriTile.Triton.Float.ScalarOps
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Codec
open VeriTile.Triton
abbrev Word := FP.Value .fp32
abbrev Byte := BitVec 8
abbrev Cell := Fin 4 → Byte
def encode (x : Word) : Cell := VeriTile.Bench.Repairs.MatmulKernel.Physical.encode x.bits
def decode (bytes : Cell) : Word := ⟨VeriTile.Bench.Repairs.MatmulKernel.Physical.decode bytes⟩
@[simp] theorem decode_encode (x : Word) : decode (encode x)=x := by
  cases x
  simp [decode,encode,VeriTile.Bench.Repairs.MatmulKernel.Physical.decode_encode]
@[simp] theorem encode_decode (bytes : Cell) : encode (decode bytes)=bytes :=
  VeriTile.Bench.Repairs.MatmulKernel.Physical.encode_decode bytes
#axiomsClean decode_encode
#axiomsClean encode_decode
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Codec
