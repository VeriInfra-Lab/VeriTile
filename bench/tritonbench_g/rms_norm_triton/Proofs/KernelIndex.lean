import bench.tritonbench_g.rms_norm_triton.Proofs.KernelBinding

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

/-- Physical admission for the selected signed-i32 ABI. These are allocation,
launch and representation conditions, never a premise about RMSNorm values.
Masked lanes must still form representable pointers; active cells occupy all
four aligned bytes. Copied inputs may alias originals or each other. -/
structure FiniteAdmission (a : Prepared α input weight entry preexisting) (normalized : List Nat) where
  rowsBound : rows input normalized < 2^31
  columnsBound : columns normalized < 2^31
  blockBound : nextPower (columns normalized) < 2^31
  rowProducts : ∀ row, row < rows input normalized → row*columns normalized < 2^31
  capacity : Nat → Nat
  inputBounds : ∀ row i, row < rows input normalized → i < columns normalized →
    a.xc.tensor.base+row*columns normalized+i < capacity a.xc.tensor.allocation
  weightBounds : ∀ i, i < columns normalized → a.wc.tensor.base+i < capacity a.wc.tensor.allocation
  outputBounds : ∀ row i, row < rows input normalized → i < columns normalized →
    a.output.tensor.base+row*columns normalized+i < capacity a.output.tensor.allocation
  byteBase : Nat → Nat
  aligned : 0 < rows input normalized →
    ∀ t ∈ [a.xc.tensor,a.wc.tensor,a.output.tensor], 4 ∣ byteBase t.allocation
  allocationEnd : ∀ t ∈ [a.xc.tensor,a.wc.tensor,a.output.tensor],
    byteBase t.allocation+4*capacity t.allocation ≤ 2^64
  inputPadded : ∀ row i, row < rows input normalized → i < nextPower (columns normalized) →
    byteBase a.xc.tensor.allocation+4*(a.xc.tensor.base+row*columns normalized+i) < 2^64
  weightPadded : ∀ i, i < nextPower (columns normalized) →
    byteBase a.wc.tensor.allocation+4*(a.wc.tensor.base+i) < 2^64
  outputPadded : ∀ row i, row < rows input normalized → i < nextPower (columns normalized) →
    byteBase a.output.tensor.allocation+4*(a.output.tensor.base+row*columns normalized+i) < 2^64
  separated : ∀ allocation, (preexisting allocation ∨ allocation = a.xc.tensor.allocation ∨
      allocation = a.wc.tensor.allocation) →
    byteBase a.output.tensor.allocation+4*capacity a.output.tensor.allocation ≤ byteBase allocation ∨
    byteBase allocation+4*capacity allocation ≤ byteBase a.output.tensor.allocation

abbrev i32 (n : Nat) : BitVec 32 := BitVec.ofNat 32 n
abbrev i64 (n : Nat) : BitVec 64 := BitVec.ofNat 64 n

def signExtend64 (x : BitVec 32) : BitVec 64 := BitVec.ofInt 64 x.toInt

def rowBits (row N : Nat) : BitVec 32 := i32 row*i32 N
def columnBits (i : Nat) : BitVec 32 := i32 i*i32 1

/-- Pointer increments follow the source: row and column products are extended
separately, after the initial in-place row pointer update. -/
def rowPointer (byteBase base row N i : Nat) : BitVec 64 :=
  i64 (byteBase+4*base)+signExtend64 (rowBits row N)*i64 4+
    signExtend64 (columnBits i)*i64 4

def weightPointer (byteBase base i : Nat) : BitVec 64 :=
  i64 (byteBase+4*base)+signExtend64 (i32 i)*i64 4

theorem i32_exact {n : Nat} (bound : n < 2^31) : (i32 n).toInt = (n : Int) := by
  have hn : (i32 n).toNat = n := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]

theorem signExtend_exact {n : Nat} (bound : n < 2^31) : signExtend64 (i32 n) = i64 n := by
  simp only [signExtend64,i32_exact bound,BitVec.ofInt_natCast]

@[simp] theorem rowBits_eq (row N : Nat) : rowBits row N = i32 (row*N) :=
  BitVec.ofNat_mul_ofNat _ _

@[simp] theorem columnBits_eq (i : Nat) : columnBits i = i32 i := by
  simp [columnBits,BitVec.ofNat_mul_ofNat]

theorem rowPointer_exact (byteBase base row N i : Nat)
    (hr : row*N < 2^31) (hi : i < 2^31)
    (hp : byteBase+4*(base+row*N+i) < 2^64) :
    (rowPointer byteBase base row N i).toNat = byteBase+4*(base+row*N+i) := by
  simp only [rowPointer,rowBits_eq,columnBits_eq,signExtend_exact hr,signExtend_exact hi,
    BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  have equal : byteBase+4*base+row*N*4+i*4 = byteBase+4*(base+row*N+i) := by omega
  rw [equal,Nat.mod_eq_of_lt hp]

theorem weightPointer_exact (byteBase base i : Nat) (hi : i < 2^31)
    (hp : byteBase+4*(base+i) < 2^64) :
    (weightPointer byteBase base i).toNat = byteBase+4*(base+i) := by
  simp only [weightPointer,signExtend_exact hi,
    BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  have equal : byteBase+4*base+i*4 = byteBase+4*(base+i) := by omega
  rw [equal,Nat.mod_eq_of_lt hp]

def IndexCorrespondence (a : Prepared α input weight entry preexisting) (normalized : List Nat)
    (f : FiniteAdmission a normalized) : Prop :=
  (∀ row i, row < rows input normalized → i < nextPower (columns normalized) →
    (∀ n ∈ [rows input normalized,columns normalized,nextPower (columns normalized),1,row,i,row*columns normalized,i*1],
      (i32 n).toInt = (n : Int)) ∧
    (rowBits row (columns normalized)).toInt = (row*columns normalized : Nat) ∧
    (columnBits i).toInt = (i : Int) ∧
    ((i32 i).slt (i32 (columns normalized)) = Bool.true ↔ i < columns normalized) ∧
    (rowPointer (f.byteBase a.xc.tensor.allocation) a.xc.tensor.base row (columns normalized) i).toNat =
      f.byteBase a.xc.tensor.allocation+4*(a.xc.tensor.base+row*columns normalized+i) ∧
    (weightPointer (f.byteBase a.wc.tensor.allocation) a.wc.tensor.base i).toNat =
      f.byteBase a.wc.tensor.allocation+4*(a.wc.tensor.base+i) ∧
    (rowPointer (f.byteBase a.output.tensor.allocation) a.output.tensor.base row (columns normalized) i).toNat =
      f.byteBase a.output.tensor.allocation+4*(a.output.tensor.base+row*columns normalized+i)) ∧
  (∀ row i, row < rows input normalized → i < columns normalized →
    a.xc.tensor.base+row*columns normalized+i < f.capacity a.xc.tensor.allocation ∧
    a.wc.tensor.base+i < f.capacity a.wc.tensor.allocation ∧
    a.output.tensor.base+row*columns normalized+i < f.capacity a.output.tensor.allocation ∧
    (∀ t ∈ [a.xc.tensor,a.wc.tensor,a.output.tensor], ∀ offset, offset < f.capacity t.allocation →
      f.byteBase t.allocation+4*offset+4 ≤ 2^64 ∧ 4 ∣ f.byteBase t.allocation+4*offset) ∧
    (∀ allocation byte lane, (preexisting allocation ∨ allocation = a.xc.tensor.allocation ∨
        allocation = a.wc.tensor.allocation) → byte < 4*f.capacity allocation → lane < 4 →
      f.byteBase a.output.tensor.allocation+4*(a.output.tensor.base+row*columns normalized+i)+lane ≠
        f.byteBase allocation+byte))

theorem index_correspondence (a : Prepared α input weight entry preexisting) (normalized : List Nat)
    (f : FiniteAdmission a normalized) : IndexCorrespondence a normalized f := by
  refine ⟨?_,?_⟩
  · intro row i hr hi
    have rp := f.rowProducts row hr
    have rb := f.rowsBound
    have cb := f.columnsBound
    have bb := f.blockBound
    have ib : i < 2^31 := hi.trans bb
    refine ⟨?_,?_,?_,?_,?_,?_,?_⟩
    · intro n member
      apply i32_exact
      simp only [List.mem_cons,List.not_mem_nil,or_false] at member
      rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> omega
    · rw [rowBits_eq]; exact i32_exact rp
    · rw [columnBits_eq]; exact i32_exact ib
    · rw [BitVec.slt_iff_toInt_lt,i32_exact ib,i32_exact cb]
      omega
    · exact rowPointer_exact _ _ _ _ _ rp ib (f.inputPadded row i hr hi)
    · exact weightPointer_exact _ _ _ ib (f.weightPadded i hi)
    · exact rowPointer_exact _ _ _ _ _ rp ib (f.outputPadded row i hr hi)
  · intro row i hr hi
    refine ⟨f.inputBounds row i hr hi,f.weightBounds i hi,f.outputBounds row i hr hi,?_,?_⟩
    · intro t member offset bound
      have endBound := f.allocationEnd t member
      exact ⟨by omega,dvd_add (f.aligned (Nat.zero_lt_of_lt hr) t member) (dvd_mul_right 4 offset)⟩
    · intro allocation byte lane existed hbyte hlane
      have bound := f.outputBounds row i hr hi
      rcases f.separated allocation existed with before | after <;> omega

/-- The finite pointer proof uses the backing stores of these same actual
Tensor views, not a second collection of hypothetical allocations. -/
def PhysicalBinding (f : FiniteAdmission a normalized) (x w xc wc y : TensorView) : Prop :=
  ∀ v ∈ [x,w,xc,wc,y], f.byteBase v.allocation = v.storageBase ∧
    f.capacity v.allocation = v.capacity

def ActualSourceAddresses (a : Prepared α input weight entry preexisting)
    (normalized : List Nat) (f : FiniteAdmission a normalized) (xc wc y : TensorView) : Prop :=
  ∀ row i, row < rows input normalized → i < nextPower (columns normalized) →
    (rowPointer (f.byteBase a.xc.tensor.allocation) a.xc.tensor.base row (columns normalized) i).toNat =
      xc.metadata.dataPtr.toNat+4*(row*columns normalized+i) ∧
    (weightPointer (f.byteBase a.wc.tensor.allocation) a.wc.tensor.base i).toNat =
      wc.metadata.dataPtr.toNat+4*i ∧
    (rowPointer (f.byteBase a.output.tensor.allocation) a.output.tensor.base row (columns normalized) i).toNat =
      y.metadata.dataPtr.toNat+4*(row*columns normalized+i)

theorem actual_source_addresses (a : Prepared α input weight entry preexisting)
    (binding : TensorBindings a x w xc wc y)
    (domain : InputDomain input weight normalized preexisting) (f : FiniteAdmission a normalized)
    (physical : PhysicalBinding f x w xc wc y) : ActualSourceAddresses a normalized f xc wc y := by
  have xb := (physical xc (by simp)).1
  have wb := (physical wc (by simp)).1
  have yb := (physical y (by simp)).1
  intro row i hr hi
  have ib : i < 2^31 := hi.trans f.blockBound
  have positive : 0 < columns normalized := by
    by_contra h
    have zero : columns normalized = 0 := by omega
    simp [zero,nextPower] at hi
  have numel : 0 < input.numel := by
    rw [shape_product domain]
    exact Nat.mul_pos (Nat.zero_lt_of_lt hr) positive
  have xp : 0 < xc.numel := by
    simpa only [← binding.inputCopy.numel,Tensor.numel,a.xc.shape] using numel
  have yp : 0 < y.numel := by rw [a.output_numel binding]; exact numel
  have wp : 0 < wc.numel := by
    simpa only [← binding.weightCopy.numel,Tensor.numel,a.wc.shape,domain.weightShape,columns] using positive
  refine ⟨?_,?_,?_⟩
  · rw [rowPointer_exact _ _ _ _ _ (f.rowProducts row hr) ib (f.inputPadded row i hr hi),
      binding.inputCopy.allocation,xb,binding.inputCopy.base,binding.inputCopy.pointer xp]
    omega
  · rw [weightPointer_exact _ _ _ ib (f.weightPadded i hi),
      binding.weightCopy.allocation,wb,binding.weightCopy.base,binding.weightCopy.pointer wp]
    omega
  · rw [rowPointer_exact _ _ _ _ _ (f.rowProducts row hr) ib (f.outputPadded row i hr hi),
      binding.output.allocation,yb,binding.output.base,binding.output.pointer yp]
    omega

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
