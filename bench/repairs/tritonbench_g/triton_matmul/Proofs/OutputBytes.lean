import bench.repairs.tritonbench_g.triton_matmul.Proofs.TensorBinding
import bench.repairs.tritonbench_g.triton_matmul.Proofs.Native
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.OutputBytes
open VeriTile.Triton FP.Structural
open FPValue (Storage stored)
open TensorBinding
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 6000000

theorem word_ext {a b : Word} (h : a.bits=b.bits) : a=b := by
  cases a; cases b; cases h; rfl

def byte (w : Word) (j : Nat) : BitVec 8 := w.bits.extractLsb' (8*j) 8
def pack (st : Storage) (w : Word) : Word := ⟨BitVec.ofNat 32 (w.bits.toNat%2^(8*width st))⟩

theorem read_nat (read : Read) (st : Storage) (ptr : Nat) :
    (readWord read st ptr).bits.toNat=(match st with
      | .half => (read ptr).toNat+256*(read (ptr+1)).toNat
      | .e4m3 => (read ptr).toNat)%4294967296 := by cases st <;> rfl
theorem pack_nat (st : Storage) (w : Word) :
    (pack st w).bits.toNat=(w.bits.toNat%2^(8*width st))%4294967296 := rfl
theorem byte_nat (w : Word) (j : Nat) :
    (byte w j).toNat=(w.bits.toNat/2^(8*j))%256 := by
  simp only [byte,BitVec.extractLsb'_toNat,Nat.shiftRight_eq_div_pow]
theorem native_nat (I : Native.Instructions) (st : Storage) (w : Word) :
    (Native.cast I st w).bits.toNat=(match st with
      | .half => (I.castHalf w).toNat | .e4m3 => (I.castE4M3 w).toNat)%4294967296 := by
  cases st with
  | half =>
    change ((I.castHalf w).setWidth 32).toNat=(I.castHalf w).toNat%4294967296
    exact BitVec.toNat_setWidth _ _
  | e4m3 =>
    change ((I.castE4M3 w).setWidth 32).toNat=(I.castE4M3 w).toNat%4294967296
    exact BitVec.toNat_setWidth _ _

theorem pack_read (read : Read) (st : Storage) (ptr : Nat) : pack st (readWord read st ptr)=readWord read st ptr := by
  apply word_ext
  apply BitVec.eq_of_toNat_eq
  rw [pack_nat,read_nat]
  have h0 := (read ptr).isLt
  have h1 := (read (ptr+1)).isLt
  cases st <;> norm_num [width] at * <;> omega

theorem pack_native (I : Native.Instructions) (st : Storage) (w : Word) : pack st (Native.cast I st w)=Native.cast I st w := by
  apply word_ext
  apply BitVec.eq_of_toNat_eq
  rw [pack_nat,native_nat]
  have h1 := (I.castHalf w).isLt
  have h2 := (I.castE4M3 w).isLt
  cases st <;> norm_num [width] at * <;> omega

theorem unpack (read : Read) (st : Storage) (ptr j : Nat) (hj : j<width st) :
    byte (readWord read st ptr) j=read (ptr+j) := by
  apply BitVec.eq_of_toNat_eq
  rw [byte_nat,read_nat]
  have h0 := (read ptr).isLt
  have h1 := (read (ptr+1)).isLt
  cases st <;> norm_num [width] at hj
  · interval_cases j <;> norm_num at * <;> omega
  · have zero : j=0 := by omega
    subst j
    norm_num at *
    omega

def store (read : Read) (st : Storage) (ptr : Nat) (w : Word) : Read := fun a =>
  if ptr≤a ∧ a<ptr+width st then byte w (a-ptr) else read a

def render (read : Read) (st : Storage) (c : Tensor) (s : State Word) : Read := fun a =>
  if c.metadata.dataPtr.toNat≤a ∧ a<c.metadata.dataPtr.toNat+width st*c.metadata.numel then
    byte (FPValue.read st (s.mem outputC ((a-c.metadata.dataPtr.toNat)/width st)))
      ((a-c.metadata.dataPtr.toNat)%width st)
  else read a

def Bound (read : Read) (st : Storage) (c : Tensor) (s : State Word) : Prop :=
  ∀ i,i<c.metadata.numel → s.mem outputC i=stored st (readWord read st (c.metadata.dataPtr.toNat+width st*i))

theorem render_at (read : Read) (st : Storage) (c : Tensor) (s : State Word)
    (i j : Nat) (hi : i<c.metadata.numel) (hj : j<width st) :
    render read st c s (c.metadata.dataPtr.toNat+width st*i+j)=byte (FPValue.read st (s.mem outputC i)) j := by
  have inside : c.metadata.dataPtr.toNat≤c.metadata.dataPtr.toNat+width st*i+j ∧
      c.metadata.dataPtr.toNat+width st*i+j<c.metadata.dataPtr.toNat+width st*c.metadata.numel := by
    cases st <;> simp only [width] at * <;> omega
  have hd : (c.metadata.dataPtr.toNat+width st*i+j-c.metadata.dataPtr.toNat)/width st=i := by
    cases st <;> simp only [width] at * <;> omega
  have hm : (c.metadata.dataPtr.toNat+width st*i+j-c.metadata.dataPtr.toNat)%width st=j := by
    cases st <;> simp only [width] at * <;> omega
  simp only [render,if_pos inside,hd,hm]

theorem render_initial (read : Read) (st : Storage) (c : Tensor) (s : State Word)
    (bound : Bound read st c s) : render read st c s=read := by
  funext a
  by_cases inside : c.metadata.dataPtr.toNat≤a ∧ a<c.metadata.dataPtr.toNat+width st*c.metadata.numel
  · let i := (a-c.metadata.dataPtr.toNat)/width st
    let j := (a-c.metadata.dataPtr.toNat)%width st
    have hi : i<c.metadata.numel := by dsimp [i]; cases st <;> simp only [width] at * <;> omega
    have hj : j<width st := Nat.mod_lt _ (by cases st <;> decide)
    have eq : a=c.metadata.dataPtr.toNat+width st*i+j := by dsimp [i,j]; cases st <;> simp only [width] at * <;> omega
    rw [eq,render_at read st c s i j hi hj,bound i hi,read_stored,unpack read st _ j hj]
  · simp only [render,if_neg inside]

theorem render_write (read : Read) (st : Storage) (c : Tensor) (s : State Word)
    (i : Nat) (hi : i<c.metadata.numel) (v : Word) :
    render read st c (s.write outputC i (stored st v))=
      store (render read st c s) st (c.metadata.dataPtr.toNat+width st*i) v := by
  funext a
  by_cases inside : c.metadata.dataPtr.toNat≤a ∧ a<c.metadata.dataPtr.toNat+width st*c.metadata.numel
  · have index : (a-c.metadata.dataPtr.toNat)/width st=i ↔
        c.metadata.dataPtr.toNat+width st*i≤a ∧ a<c.metadata.dataPtr.toNat+width st*i+width st := by
      cases st <;> simp only [width] at * <;> omega
    by_cases hit : (a-c.metadata.dataPtr.toNat)/width st=i
    · have remain : (a-c.metadata.dataPtr.toNat)%width st=a-(c.metadata.dataPtr.toNat+width st*i) := by
        cases st <;> simp only [width] at * <;> omega
      simp only [render,if_pos inside,State.write,hit,and_self,ite_true,read_stored,
        store,if_pos (index.mp hit),remain]
    · simp only [render,if_pos inside,State.write,true_and,if_neg hit,
        store,if_neg (index.not.mp hit)]
  · have miss : ¬(c.metadata.dataPtr.toNat+width st*i≤a ∧ a<c.metadata.dataPtr.toNat+width st*i+width st) := by
      cases st <;> simp only [width] at * <;> omega
    simp only [render,if_neg inside,store,if_neg miss]

theorem render_read (read : Read) (st : Storage) (c : Tensor) (s : State Word)
    (i : Nat) (hi : i<c.metadata.numel) :
    readWord (render read st c s) st (c.metadata.dataPtr.toNat+width st*i)=pack st (FPValue.read st (s.mem outputC i)) := by
  apply word_ext
  apply BitVec.eq_of_toNat_eq
  rw [read_nat,pack_nat]
  have h0 := render_at read st c s i 0 hi (by cases st <;> decide)
  simp only [Nat.add_zero] at h0
  cases st with
  | half =>
    have h1 := render_at read .half c s i 1 hi (by decide)
    simp only [width] at h0 h1
    simp only [width,h0,h1,byte_nat]
    norm_num
    omega
  | e4m3 =>
    simp only [width,Nat.one_mul] at h0
    simp only [width,Nat.one_mul,h0,byte_nat]
    norm_num

theorem frame (read : Read) (st : Storage) (c : Tensor) (s : State Word)
    (a : Nat) (outside : a<c.metadata.dataPtr.toNat ∨ c.metadata.dataPtr.toNat+width st*c.metadata.numel≤a) :
    render read st c s a=read a := by
  simp only [render,if_neg (by omega : ¬(c.metadata.dataPtr.toNat≤a ∧ a<c.metadata.dataPtr.toNat+width st*c.metadata.numel))]

#axiomsClean pack_native
#axiomsClean render_initial
#axiomsClean render_write
#axiomsClean render_read
#axiomsClean frame
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.OutputBytes
