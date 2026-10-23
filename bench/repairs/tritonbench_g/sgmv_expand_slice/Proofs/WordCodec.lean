import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.CallerLogical
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.NativeGraph
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.NativeBackend
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostPrepared

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.WordCodec

def bytes16 (word : BitVec 16) : StoredWord 2 :=
  fun b => if b.val = 0 then word.extractLsb' 0 8 else word.extractLsb' 8 8
def read16 (bytes : StoredWord 2) : BitVec 16 :=
  (bytes ⟨1,by decide⟩).append (bytes ⟨0,by decide⟩)

def bytes32 (word : BitVec 32) : StoredWord 4 := fun b =>
  if b.val = 0 then word.extractLsb' 0 8 else if b.val = 1 then word.extractLsb' 8 8
  else if b.val = 2 then word.extractLsb' 16 8 else word.extractLsb' 24 8
def read32 (bytes : StoredWord 4) : BitVec 32 :=
  (bytes ⟨3,by decide⟩).append ((bytes ⟨2,by decide⟩).append
    ((bytes ⟨1,by decide⟩).append (bytes ⟨0,by decide⟩)))

theorem bytes16_read (word : BitVec 16) : read16 (bytes16 word) = word := by
  simp only [read16,bytes16]
  bv_decide
theorem bytes32_read (word : BitVec 32) : read32 (bytes32 word) = word := by
  simp only [read32,bytes32]
  bv_decide

def sign16 (negative : Bool) (word : BitVec 16) := if negative then word ^^^ 0x8000 else word
def sign32 (negative : Bool) (word : BitVec 32) := if negative then word ^^^ 0x80000000 else word

theorem sign16_twice (negative : Bool) (word : BitVec 16) : sign16 negative (sign16 negative word) = word := by
  cases negative <;> simp [sign16, BitVec.xor_assoc]
theorem sign32_twice (negative : Bool) (word : BitVec 32) : sign32 negative (sign32 negative word) = word := by
  cases negative <;> simp [sign32, BitVec.xor_assoc]

/-- An inhabited explicit per-element bit representation for the logical view.
Native tensor-library conformance uses its ordinary local view-copy contract;
this bit identity is not a claim of GPU execution or conformance. -/
def codec16 (view : OutputView) (width : view.width = 2) : CallerLogical.Codec (BitVec 16) view where
  encode := fun word b => bytes16 (sign16 view.negative word) ⟨b.val,by have := b.isLt; omega⟩
  decode := fun bytes => sign16 view.negative (read16 (fun b => bytes ⟨b.val,by have := b.isLt; omega⟩))
  read_write := by
    intro word
    change sign16 view.negative (read16 (bytes16 (sign16 view.negative word))) = word
    rw [bytes16_read,sign16_twice]

def codec32 (view : OutputView) (width : view.width = 4) : CallerLogical.Codec (BitVec 32) view where
  encode := fun word b => bytes32 (sign32 view.negative word) ⟨b.val,by have := b.isLt; omega⟩
  decode := fun bytes => sign32 view.negative (read32 (fun b => bytes ⟨b.val,by have := b.isLt; omega⟩))
  read_write := by
    intro word
    change sign32 view.negative (read32 (bytes32 (sign32 view.negative word))) = word
    rw [bytes32_read,sign32_twice]

/-- The output pointer ABI chooses the actual two/four-byte codec. -/
def forABI (abi : SourceABI) (view : OutputView) (width : view.width = HostPrepared.scalarBytes abi.output) :
    CallerLogical.Codec (Native.Word (NativeBackend.profile abi).output) view := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases output with
  | half => exact codec16 view width
  | bfloat => exact codec16 view width
  | single => exact codec32 view width

#axiomsClean bytes16_read
#axiomsClean bytes32_read
#axiomsClean sign16_twice
#axiomsClean sign32_twice
#axiomsClean codec16
#axiomsClean codec32
#axiomsClean forABI
end VeriTile.Bench.Repairs.SgmvExpandSlice.WordCodec
