import bench.repairs.tritonbench_g.triton_argmax.revision004.Proofs.SelectedEmission

/-! Exact bit transport. These equations concern all word patterns and do not
use floating operations, integer value ranges or a numerical conformance port. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Revision004.NativePayload

def low (word : BitVec 64) : BitVec 32 := word.extractLsb' 0 32
def high (word : BitVec 64) : BitVec 32 := word.extractLsb' 32 32
def join (lo hi : BitVec 32) : BitVec 64 :=
  lo.setWidth 64 ||| (hi.setWidth 64 <<< 32)

theorem reassemble (word : BitVec 64) : join (low word) (high word)=word := by
  unfold join low high
  rw [BitVec.or_comm]
  rw [← BitVec.setWidth_append_eq_shiftLeft_setWidth_or]
  simpa only [BitVec.setWidth_eq] using (BitVec.extractLsb'_append_extractLsb' (x:=word) (w:=32) (len:=32))

def sharedWord (word : BitVec bits) : BitVec bits := BitVec.ofNat bits word.toNat

theorem sharedWord_exact (word : BitVec bits) : sharedWord word=word := by
  simp [sharedWord]

#axiomsClean reassemble
#axiomsClean sharedWord_exact
end VeriTile.Bench.Repairs.TritonArgmax.Revision004.NativePayload
