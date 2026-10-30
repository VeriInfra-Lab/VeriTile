import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.Native

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.Addresses
open VeriTile.Triton FP.Structural FPValue Host TensorABI Native
noncomputable section

def Dimensions (h : Shape) : Prop := h.rows<2^31 ∧ h.columns<2^31 ∧ h.inner<2^31
theorem observed_dimensions (h : Shape) (f : Flags) (cfg : Config) (profile : Profile)
    (scope : ObservedSchedule h f cfg profile) : Dimensions h := by
  cases profile <;> rcases scope with ⟨rfl,_⟩ <;> norm_num [Dimensions]

abbrev word (n : Nat) : BitVec 64 := BitVec.ofNat 64 n
def lane (n : Nat) : BitVec 64 := (BitVec.ofNat 32 n).signExtend 64
/-- Binder specialization of nonnegative runtime strides: one is constexpr;
otherwise the observed profiles use signed32 widened before multiplication. -/
def scalar (n : Nat) : BitVec 64 := if n=1 then word 1 else lane n
theorem lane_word (n : Nat) (bound : n<2^31) : lane n=word n := by
  have hn : (BitVec.ofNat 32 n).toNat=n := by
    simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (show n<2^32 by omega)]
  have hi : (BitVec.ofNat 32 n).toInt=(n:Int) := by
    rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]
  change BitVec.ofInt 64 (BitVec.ofNat 32 n).toInt=word n
  rw [hi,BitVec.ofInt_natCast]
theorem scalar_word (n : Nat) (bound : n<2^31) : scalar n=word n := by
  unfold scalar
  split_ifs with h
  · subst n; rfl
  · exact lane_word n bound

def offset (i j sr sc : Nat) : BitVec 64 := lane i*scalar sr+lane j*scalar sc
def address (base bytes i j sr sc : Nat) : BitVec 64 := word base+word bytes*offset i j sr sc
theorem address_pointer (base bytes i j sr sc : Nat) (hi : i<2^31) (hj : j<2^31)
    (hr : sr<2^31) (hc : sc<2^31) :
    address base bytes i j sr sc=pointer base bytes (i*sr+j*sc) := by
  simp only [address,offset,lane_word _ hi,lane_word _ hj,scalar_word _ hr,scalar_word _ hc,
    BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  rfl

def aPointer (h : Shape) (f : Flags) (a : TensorABI.Tensor) (i : AIndex (params h f)) :=
  address a.metadata.dataPtr.toNat 2 i.1.val i.2.1.val (params h f).aRow (params h f).aInner
def bPointer (h : Shape) (f : Flags) (b : TensorABI.Tensor) (i : BIndex (params h f)) :=
  address b.metadata.dataPtr.toNat 2 i.1.val i.2.1.val (params h f).bInner (params h f).bColumn
def cPointer (h : Shape) (f : Flags) (c : TensorABI.Tensor) (i : CIndex (params h f)) :=
  address c.metadata.dataPtr.toNat (width f.half) i.1.val i.2.1.val h.columns 1

theorem a_pointer (h : Shape) (f : Flags) (dim : Dimensions h) (a : TensorABI.Tensor) (i : AIndex (params h f)) :
    aPointer h f a i=pointer a.metadata.dataPtr.toNat 2 (aOffset (params h f) i) := by
  apply address_pointer _ _ _ _ _ _ (lt_trans i.1.isLt dim.1) (lt_trans i.2.1.isLt dim.2.2)
  all_goals have hm := dim.1; have hk := dim.2.2; cases hf : f.transA <;> simp [params,parameters,inputA,contiguous,transpose,hf] <;> omega

theorem b_pointer (h : Shape) (f : Flags) (dim : Dimensions h) (b : TensorABI.Tensor) (i : BIndex (params h f)) :
    bPointer h f b i=pointer b.metadata.dataPtr.toNat 2 (bOffset (params h f) i) := by
  apply address_pointer _ _ _ _ _ _ (lt_trans i.1.isLt dim.2.2) (lt_trans i.2.1.isLt dim.2.1)
  all_goals have hn := dim.2.1; have hk := dim.2.2; cases hf : f.transB <;> simp [params,parameters,inputB,contiguous,transpose,hf] <;> omega

theorem c_pointer (h : Shape) (f : Flags) (dim : Dimensions h) (c : TensorABI.Tensor) (i : CIndex (params h f)) :
    cPointer h f c i=pointer c.metadata.dataPtr.toNat (width f.half) (cOffset (params h f) i) := by
  rw [show cOffset (params h f) i=i.1.val*h.columns+i.2.1.val from c_offset h f regions i]
  simpa only [Nat.mul_one] using address_pointer c.metadata.dataPtr.toNat (width f.half)
    i.1.val i.2.1.val h.columns 1 (lt_trans i.1.isLt dim.1) (lt_trans i.2.1.isLt dim.2.1) dim.2.1 (by decide)

theorem all_coordinates (h : Shape) (f : Flags) (dim : Dimensions h)
    (a b c : TensorABI.Tensor) (abi : Binding h f a b c) :
    (∀ i, (aPointer h f a i).toNat=physicalAddress 2 a i.1.val i.2.1.val) ∧
    (∀ i, (bPointer h f b i).toNat=physicalAddress 2 b i.1.val i.2.1.val) ∧
    (∀ i, (cPointer h f c i).toNat=physicalAddress (width f.half) c i.1.val i.2.1.val) := by
  simpa only [a_pointer h f dim,b_pointer h f dim,c_pointer h f dim] using source_addresses h f a b c abi

#axiomsClean all_coordinates
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.Addresses
