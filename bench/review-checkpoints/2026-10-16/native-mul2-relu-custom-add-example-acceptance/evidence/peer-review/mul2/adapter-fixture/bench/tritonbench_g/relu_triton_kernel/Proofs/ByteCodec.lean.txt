import bench.tritonbench_g.relu_triton_kernel.AddressGridCorrect

/-! Actual little-endian byte assembly and splitting over mathematical memory.
Each address contains exactly eight bits. Contiguous bytes are concatenated,
not re-labelled as a wide logical cell. Native storage/provenance is separate. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.ByteMemory
open VeriTile.Triton FP.Structural IntegerWords FiniteAddress

abbrev Byte := BitVec 8
abbrev Memory := Nat → Byte

/-- Low-address bytes occupy low-order bits. -/
def loadLE : (count : Nat) → Memory → Nat → BitVec (count * 8)
  | 0, _, _ => 0
  | count + 1, mem, base =>
    ((loadLE count mem (base + 1)) ++ mem base).cast (by omega)

def splitByte {bits : Nat} (word : BitVec bits) (byte : Nat) : Byte :=
  word.extractLsb' (byte * 8) 8

def storeLE (count : Nat) (mem : Memory) (base : Nat)
    (word : BitVec (count * 8)) : Memory :=
  fun address => if base ≤ address ∧ address < base + count then
    splitByte word (address - base) else mem address

theorem load_bit (count : Nat) (mem : Memory) (base bit : Nat) (within : bit < count * 8) :
    (loadLE count mem base).getLsbD bit = (mem (base + bit / 8)).getLsbD (bit % 8) := by
  induction count generalizing base bit with
  | zero => omega
  | succ count ih =>
    simp only [loadLE, BitVec.getLsbD_cast, BitVec.getLsbD_append]
    by_cases low : bit < 8
    · simp [low, Nat.div_eq_of_lt low, Nat.mod_eq_of_lt low]
    · simp only [low, ↓reduceIte]
      rw [ih (base + 1) (bit - 8) (by omega)]
      rw [show base + 1 + (bit - 8) / 8 = base + bit / 8 by omega,
        show (bit - 8) % 8 = bit % 8 by omega]

theorem load_congr (count : Nat) (left right : Memory) (base : Nat)
    (same : ∀ byte, byte < count → left (base + byte) = right (base + byte)) :
    loadLE count left base = loadLE count right base := by
  apply BitVec.eq_of_getLsbD_eq
  intro bit within
  rw [load_bit count left base bit within, load_bit count right base bit within,
    same (bit / 8) (by omega)]

theorem store_byte (count : Nat) (mem : Memory) (base : Nat)
    (word : BitVec (count * 8)) (byte : Nat) (within : byte < count) :
    storeLE count mem base word (base + byte) = splitByte word byte := by
  simp [storeLE, show base + byte < base + count by omega]

theorem store_frame (count : Nat) (mem : Memory) (base : Nat)
    (word : BitVec (count * 8)) (address : Nat)
    (outside : address < base ∨ base + count ≤ address) :
    storeLE count mem base word address = mem address := by
  unfold storeLE
  rw [if_neg]
  omega

theorem load_store (count : Nat) (mem : Memory) (base : Nat)
    (word : BitVec (count * 8)) : loadLE count (storeLE count mem base word) base = word := by
  apply BitVec.eq_of_getLsbD_eq
  intro bit within
  rw [load_bit count _ base bit within, store_byte count mem base word (bit / 8) (by omega)]
  simp only [splitByte, BitVec.getLsbD_extractLsb']
  have low : bit % 8 < 8 := Nat.mod_lt _ (by decide)
  simp only [low, decide_true, Bool.true_and]
  congr 1
  omega

theorem split_load (count : Nat) (mem : Memory) (base byte : Nat) (within : byte < count) :
    splitByte (loadLE count mem base) byte = mem (base + byte) := by
  apply BitVec.eq_of_getLsbD_eq
  intro bit low
  simp only [splitByte, BitVec.getLsbD_extractLsb', low, decide_true, Bool.true_and]
  rw [load_bit count mem base (byte * 8 + bit) (by omega)]
  congr 2 <;> omega

theorem store_load (count : Nat) (mem : Memory) (base : Nat) :
    storeLE count mem base (loadLE count mem base) = mem := by
  funext address
  by_cases active : base ≤ address ∧ address < base + count
  · rw [show address = base + (address - base) by omega,
      store_byte count mem base (loadLE count mem base) (address - base) (by omega),
      split_load count mem base (address - base) (by omega)]
  · unfold storeLE
    rw [if_neg active]

def loadInput (encoding : Encoding) (mem : Memory) (base offset : Nat) : Word encoding :=
  (loadLE (elementBytes encoding) mem (base + offset * elementBytes encoding)).cast
    (element_bytes_width encoding)

def load32 (mem : Memory) (base offset : Nat) : BitVec 32 :=
  loadLE 4 mem (base + offset * 4)

def storeInput (encoding : Encoding) (mem : Memory) (base offset : Nat)
    (word : Word encoding) : Memory :=
  storeLE (elementBytes encoding) mem (base + offset * elementBytes encoding)
    (word.cast (element_bytes_width encoding).symm)

theorem input_word_roundtrip (encoding : Encoding) (mem : Memory) (base offset : Nat)
    (word : Word encoding) :
    loadInput encoding (storeInput encoding mem base offset word) base offset = word := by
  unfold loadInput storeInput
  rw [load_store]
  rfl

theorem input_bytes_roundtrip (encoding : Encoding) (mem : Memory) (base offset : Nat) :
    storeInput encoding mem base offset (loadInput encoding mem base offset) = mem := by
  unfold storeInput loadInput
  exact store_load (elementBytes encoding) mem (base + offset * elementBytes encoding)

def loadAt64 (count : Nat) (mem : Memory) (address : BitVec 64) : BitVec (count * 8) :=
  loadLE count mem address.toNat

def storeAt64 (count : Nat) (mem : Memory) (address : BitVec 64)
    (word : BitVec (count * 8)) : Memory := storeLE count mem address.toNat word

theorem finite_pointer_load (count base N pid lane : Nat) (mem : Memory)
    (indexDomain : IndexDomain N) (domain : ByteDomain base count N)
    (positive : 0 < count) (launched : pid < GatedGrid.gridCount N blockSize)
    (inBlock : lane < blockSize) (active : offset pid lane < N) :
    loadAt64 count mem (pointer64 base count pid lane) =
      loadLE count mem (base + offset pid lane * count) := by
  unfold loadAt64
  rw [pointer_exact base count N pid lane indexDomain domain positive launched inBlock active]

theorem finite_pointer_store (count base N pid lane : Nat) (mem : Memory)
    (word : BitVec (count * 8)) (indexDomain : IndexDomain N) (domain : ByteDomain base count N)
    (positive : 0 < count) (launched : pid < GatedGrid.gridCount N blockSize)
    (inBlock : lane < blockSize) (active : offset pid lane < N) :
    storeAt64 count mem (pointer64 base count pid lane) word =
      storeLE count mem (base + offset pid lane * count) word := by
  unfold storeAt64
  rw [pointer_exact base count N pid lane indexDomain domain positive launched inBlock active]

#specNonCircular loadLE avoiding [TypedValue.source, GatedGrid.runGrid, FP.Structural.exec]
#axiomsClean load_bit
#axiomsClean load_congr
#axiomsClean store_byte
#axiomsClean store_frame
#axiomsClean load_store
#axiomsClean split_load
#axiomsClean store_load
#axiomsClean input_word_roundtrip
#axiomsClean input_bytes_roundtrip
#axiomsClean finite_pointer_load
#axiomsClean finite_pointer_store
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.ByteMemory
