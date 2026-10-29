import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.Native

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.Bytes
open VeriTile.Triton FP.Structural FPValue Host TensorABI Native
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
abbrev Read := Nat → BitVec 8

def readWord (read : Read) (half : Bool) (ptr : Nat) : Word := ⟨BitVec.ofNat 32
  (if half then (read ptr).toNat+256*(read (ptr+1)).toNat else
    (read ptr).toNat+256*(read (ptr+1)).toNat+65536*(read (ptr+2)).toNat+16777216*(read (ptr+3)).toNat)⟩
def byte (w : Word) (j : Nat) : BitVec 8 := w.bits.extractLsb' (8*j) 8
def pack (half : Bool) (w : Word) : Word := ⟨BitVec.ofNat 32 (w.bits.toNat%2^(8*width half))⟩
theorem word_ext {a b : Word} (h : a.bits=b.bits) : a=b := by cases a; cases b; cases h; rfl
theorem read_nat (read : Read) (half : Bool) (ptr : Nat) :
    (readWord read half ptr).bits.toNat = (if half then (read ptr).toNat+256*(read (ptr+1)).toNat else
      (read ptr).toNat+256*(read (ptr+1)).toNat+65536*(read (ptr+2)).toNat+16777216*(read (ptr+3)).toNat)%4294967296 := rfl
theorem pack_nat (half : Bool) (w : Word) :
    (pack half w).bits.toNat=(w.bits.toNat%2^(8*width half))%4294967296 := rfl
theorem byte_nat (w : Word) (j : Nat) : (byte w j).toNat=(w.bits.toNat/2^(8*j))%256 := by
  simp only [byte,BitVec.extractLsb'_toNat,Nat.shiftRight_eq_div_pow]

theorem pack_read (read : Read) (half : Bool) (ptr : Nat) : pack half (readWord read half ptr)=readWord read half ptr := by
  apply word_ext
  apply BitVec.eq_of_toNat_eq
  rw [pack_nat,read_nat]
  have h0 := (read ptr).isLt
  have h1 := (read (ptr+1)).isLt
  cases half <;> norm_num [width] at * <;> omega

theorem cast_nat (I : Instructions) (half : Bool) (w : Word) :
    (Native.cast I half w).bits.toNat = (if half then (I.castHalf w).toNat%4294967296 else w.bits.toNat) := by
  cases half
  · rfl
  · change ((I.castHalf w).setWidth 32).toNat=(I.castHalf w).toNat%4294967296
    exact BitVec.toNat_setWidth _ _

theorem pack_native (I : Instructions) (half : Bool) (w : Word) : pack half (Native.cast I half w)=Native.cast I half w := by
  apply word_ext
  apply BitVec.eq_of_toNat_eq
  rw [pack_nat,cast_nat]
  have h1 := (I.castHalf w).isLt
  have hw : w.bits.toNat < 4294967296 := w.bits.isLt
  cases half <;> norm_num [width] at * <;> omega

theorem unpack (read : Read) (half : Bool) (ptr j : Nat) (hj : j<width half) :
    byte (readWord read half ptr) j=read (ptr+j) := by
  apply BitVec.eq_of_toNat_eq
  rw [byte_nat,read_nat]
  have h0 := (read ptr).isLt
  have h1 := (read (ptr+1)).isLt
  have h2 := (read (ptr+2)).isLt
  have h3 := (read (ptr+3)).isLt
  cases half <;> norm_num [width] at hj ⊢
  all_goals interval_cases j <;> norm_num at * <;> omega

def store (read : Read) (half : Bool) (ptr : Nat) (w : Word) : Read := fun a =>
  if ptr≤a ∧ a<ptr+width half then byte w (a-ptr) else read a

def render (read : Read) (half : Bool) (c : TensorABI.Tensor) (count : Nat) (s : State Word) : Read := fun a =>
  if c.metadata.dataPtr.toNat≤a ∧ a<c.metadata.dataPtr.toNat+width half*count then
    byte (value (outputDType half) (s.mem "C" ((a-c.metadata.dataPtr.toNat)/width half)))
      ((a-c.metadata.dataPtr.toNat)%width half)
  else read a

def Bound (read : Read) (half : Bool) (c : TensorABI.Tensor) (count : Nat) (s : State Word) : Prop :=
  ∀ i,i<count → s.mem "C" i=stored (outputDType half) (readWord read half (c.metadata.dataPtr.toNat+width half*i))

theorem render_at (read : Read) (half : Bool) (c : TensorABI.Tensor) (count : Nat) (s : State Word)
    (i j : Nat) (hi : i<count) (hj : j<width half) :
    render read half c count s (c.metadata.dataPtr.toNat+width half*i+j)=byte (value (outputDType half) (s.mem "C" i)) j := by
  have inside : c.metadata.dataPtr.toNat≤c.metadata.dataPtr.toNat+width half*i+j ∧
      c.metadata.dataPtr.toNat+width half*i+j<c.metadata.dataPtr.toNat+width half*count := by
    cases half <;> simp only [width,Bool.false_eq_true,↓reduceIte] at * <;> omega
  have hd : (c.metadata.dataPtr.toNat+width half*i+j-c.metadata.dataPtr.toNat)/width half=i := by
    cases half <;> simp only [width,Bool.false_eq_true,↓reduceIte] at * <;> omega
  have hm : (c.metadata.dataPtr.toNat+width half*i+j-c.metadata.dataPtr.toNat)%width half=j := by
    cases half <;> simp only [width,Bool.false_eq_true,↓reduceIte] at * <;> omega
  simp only [render,if_pos inside,hd,hm]

theorem render_initial (read : Read) (half : Bool) (c : TensorABI.Tensor) (count : Nat) (s : State Word)
    (bound : Bound read half c count s) : render read half c count s=read := by
  funext a
  by_cases inside : c.metadata.dataPtr.toNat≤a ∧ a<c.metadata.dataPtr.toNat+width half*count
  · let i := (a-c.metadata.dataPtr.toNat)/width half
    let j := (a-c.metadata.dataPtr.toNat)%width half
    have hi : i<count := by dsimp [i]; cases half <;> simp only [width,Bool.false_eq_true,↓reduceIte] at * <;> omega
    have hj : j<width half := Nat.mod_lt _ (by cases half <;> decide)
    have eq : a=c.metadata.dataPtr.toNat+width half*i+j := by dsimp [i,j]; cases half <;> simp only [width,Bool.false_eq_true,↓reduceIte] at * <;> omega
    rw [eq,render_at read half c count s i j hi hj,bound i hi,read_stored,unpack read half _ j hj]
  · simp only [render,if_neg inside]

theorem render_write (read : Read) (half : Bool) (c : TensorABI.Tensor) (count : Nat) (s : State Word)
    (i : Nat) (hi : i<count) (v : Word) :
    render read half c count (s.write "C" i (stored (outputDType half) v))=
      store (render read half c count s) half (c.metadata.dataPtr.toNat+width half*i) v := by
  funext a
  by_cases inside : c.metadata.dataPtr.toNat≤a ∧ a<c.metadata.dataPtr.toNat+width half*count
  · have index : (a-c.metadata.dataPtr.toNat)/width half=i ↔
        c.metadata.dataPtr.toNat+width half*i≤a ∧ a<c.metadata.dataPtr.toNat+width half*i+width half := by
      cases half <;> simp only [width,Bool.false_eq_true,↓reduceIte] at * <;> omega
    by_cases hit : (a-c.metadata.dataPtr.toNat)/width half=i
    · have remain : (a-c.metadata.dataPtr.toNat)%width half=a-(c.metadata.dataPtr.toNat+width half*i) := by
        cases half <;> simp only [width,Bool.false_eq_true,↓reduceIte] at * <;> omega
      simp only [render,if_pos inside,State.write,hit,and_self,ite_true,read_stored,
        store,if_pos (index.mp hit),remain]
    · simp only [render,if_pos inside,State.write,true_and,if_neg hit,
        store,if_neg (index.not.mp hit)]
  · have miss : ¬(c.metadata.dataPtr.toNat+width half*i≤a ∧ a<c.metadata.dataPtr.toNat+width half*i+width half) := by
      cases half <;> simp only [width,Bool.false_eq_true,↓reduceIte] at * <;> omega
    simp only [render,if_neg inside,store,if_neg miss]

theorem render_read (read : Read) (half : Bool) (c : TensorABI.Tensor) (count : Nat) (s : State Word)
    (i : Nat) (hi : i<count) :
    readWord (render read half c count s) half (c.metadata.dataPtr.toNat+width half*i)=pack half (value (outputDType half) (s.mem "C" i)) := by
  apply word_ext
  apply BitVec.eq_of_toNat_eq
  rw [read_nat,pack_nat]
  have h0 := render_at read half c count s i 0 hi (by cases half <;> decide)
  simp only [Nat.add_zero] at h0
  cases half with
  | «false» =>
    have h1 := render_at read Bool.false c count s i 1 hi (by decide)
    have h2 := render_at read Bool.false c count s i 2 hi (by decide)
    have h3 := render_at read Bool.false c count s i 3 hi (by decide)
    simp only [width,Bool.false_eq_true,↓reduceIte] at h0 h1 h2 h3
    simp only [width,Bool.false_eq_true,↓reduceIte,h0,h1,h2,h3,byte_nat]
    have hv : (value (outputDType Bool.false) (s.mem "C" i)).bits.toNat < 4294967296 :=
      (value (outputDType Bool.false) (s.mem "C" i)).bits.isLt
    norm_num at *
    omega
  | «true» =>
    have h1 := render_at read Bool.true c count s i 1 hi (by decide)
    simp only [width,↓reduceIte] at h0 h1
    simp only [width,↓reduceIte,h0,h1,byte_nat]
    norm_num
    omega

theorem frame (read : Read) (half : Bool) (c : TensorABI.Tensor) (count : Nat) (s : State Word)
    (a : Nat) (outside : a<c.metadata.dataPtr.toNat ∨ c.metadata.dataPtr.toNat+width half*count≤a) :
    render read half c count s a=read a := by
  simp only [render,if_neg (by omega : ¬(c.metadata.dataPtr.toNat≤a ∧ a<c.metadata.dataPtr.toNat+width half*count))]


#axiomsClean pack_native
#axiomsClean render_initial
#axiomsClean render_write
#axiomsClean render_read
#axiomsClean frame
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.Bytes
