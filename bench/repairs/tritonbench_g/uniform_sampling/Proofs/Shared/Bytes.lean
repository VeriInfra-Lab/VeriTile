import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Memory
import VeriTile.Triton.Float.BitValue

namespace VeriTile.Bench.Repairs.UniformSampling.Shared.Bytes
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
open VeriTile.Triton
noncomputable section
open scoped Classical
abbrev Byte := BitVec 8
abbrev Store := Nat→Byte

def encode (word : Float32Bits) (i : Fin 4) : Byte :=
  word.bits.extractLsb' (8*i.val) 8

def decode (bytes : Fin 4→Byte) : Float32Bits :=
  ⟨((bytes 3 ++ bytes 2) ++ bytes 1) ++ bytes 0⟩

theorem decode_encode (word : Float32Bits) : decode (encode word)=word := by
  cases word with | mk word =>
    apply congrArg Float32Bits.mk
    dsimp [decode,encode]
    bv_decide

theorem encode_decode (bytes : Fin 4→Byte) : encode (decode bytes)=bytes := by
  funext i
  fin_cases i
  · change (((bytes 3 ++ bytes 2) ++ bytes 1) ++ bytes 0).extractLsb' 0 8=bytes 0
    simp only [BitVec.extractLsb'_append_eq_right]
  · change (((bytes 3 ++ bytes 2) ++ bytes 1) ++ bytes 0).extractLsb' 8 8=bytes 1
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8 ≤ 8)]
    exact BitVec.extractLsb'_append_eq_right
  · change (((bytes 3 ++ bytes 2) ++ bytes 1) ++ bytes 0).extractLsb' 16 8=bytes 2
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8 ≤ 16)]
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8 ≤ 8)]
    exact BitVec.extractLsb'_append_eq_right
  · change (((bytes 3 ++ bytes 2) ++ bytes 1) ++ bytes 0).extractLsb' 24 8=bytes 3
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8 ≤ 24)]
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8 ≤ 16)]
    exact BitVec.extractLsb'_append_eq_left

def read (m : Store) (a : Nat) : Float32Bits := decode (fun i=>m (a+i.val))
def write (m : Store) (a : Nat) (v : Float32Bits) : Store :=
  fun q=>if h:a≤q ∧ q<a+4 then encode v ⟨q-a,by omega⟩ else m q

def api : API Float32Bits Store Byte where
  read:=read
  write:=write
  byte:=fun m=>m
  read_same:=by
    intro m a v
    have h : (fun i : Fin 4=>write m a v (a+i.val))=encode v := by
      funext i
      have hi:=i.isLt
      simp [write,show a≤a+i.val ∧ a+i.val<a+4 by omega]
    simpa only [read,h] using decode_encode v
  read_apart:=by
    intro m a b v h
    apply congrArg decode
    funext i
    have hi:=i.isLt
    unfold Apart at h
    simp [write,show ¬(a≤b+i.val ∧ b+i.val<a+4) by omega]
  byte_frame:=by
    intro m a i v h
    simp [write,show ¬(a ≤ i ∧ i < a+4) by omega]

theorem byte_of_read (m : Store) (a : Nat) (i : Fin 4) :
    api.byte m (a+i.val)=encode (api.read m a) i := by
  exact (congrFun (encode_decode (fun j=>m (a+j.val))) i).symm

#axiomsClean api
#axiomsClean byte_of_read
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.Shared.Bytes
