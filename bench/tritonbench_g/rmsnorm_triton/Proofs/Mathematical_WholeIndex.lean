import bench.tritonbench_g.rmsnorm_triton.Proofs.Mathematical_WholeHost

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 10000

/-- Public finite support for the audited signed-i32 stride ABI. Masked lanes
still form all addresses. Input row and column offsets are extended separately;
the output's three terms are summed in i32 before pointer extension. -/
structure FiniteAdmission (p : Parameters) (d : Domain p) (preexisting : Nat → Prop) where
  loopEnd : trips p*1024 < 2^31
  gridX : p.batches < 2^31
  gridY : p.rows ≤ 65535
  inputStrides : ∀ ax, ax < 3 → p.input.stride ax < 2^31
  weightStride : p.weights.stride 0 < 2^31
  outputStrides : ∀ ax, ax < 3 → p.output.stride ax < 2^31
  inputRow : ∀ b m, b < p.batches → m < p.rows →
    b*p.input.stride 0+m*p.input.stride 1 < 2^31
  inputColumn : ∀ k, k < trips p*1024 → k*p.input.stride 2 < 2^31
  weightColumn : ∀ k, k < trips p*1024 → k*p.weights.stride 0 < 2^31
  outputCombined : ∀ b m k, b < p.batches → m < p.rows → k < trips p*1024 →
    b*p.output.stride 0+m*p.output.stride 1+k*p.output.stride 2 < 2^31
  byteBase : Nat → Nat
  aligned : ∀ t ∈ [p.input,p.weights,p.output], 4 ∣ byteBase t.allocation
  allocationEnd : ∀ t ∈ [p.input,p.weights,p.output],
    byteBase t.allocation+4*d.capacity t.allocation ≤ 2^64
  inputPadded : ∀ b m k, b < p.batches → m < p.rows → k < trips p*1024 →
    byteBase p.input.allocation+4*inputOffset p b m k < 2^64
  weightPadded : ∀ k, k < trips p*1024 →
    byteBase p.weights.allocation+4*weightOffset p k < 2^64
  outputPadded : ∀ b m k, b < p.batches → m < p.rows → k < trips p*1024 →
    byteBase p.output.allocation+4*outputOffset p b m k < 2^64
  preexistingSeparated : ∀ a, preexisting a →
    byteBase p.output.allocation+4*d.capacity p.output.allocation ≤ byteBase a ∨
    byteBase a+4*d.capacity a ≤ byteBase p.output.allocation

abbrev i32 (n : Nat) : BitVec 32 := BitVec.ofNat 32 n
abbrev i64 (n : Nat) : BitVec 64 := BitVec.ofNat 64 n

def loopBits (t : Nat) : BitVec 32 := i32 t * i32 1024
def columnBits (t j : Nat) : BitVec 32 := loopBits t+i32 j
def rowBits (x : Tensor) (b m : Nat) : BitVec 32 :=
  i32 b*i32 (x.stride 0)+i32 m*i32 (x.stride 1)
def colProductBits (x : Tensor) (ax t j : Nat) : BitVec 32 :=
  columnBits t j*i32 (x.stride ax)
def outputBits (p : Parameters) (b m t j : Nat) : BitVec 32 :=
  rowBits p.output b m+colProductBits p.output 2 t j

/-- Signed extension used by pointer arithmetic, expressed via the signed
integer interpretation rather than an unsigned cast. -/
def signExtend64 (x : BitVec 32) : BitVec 64 := BitVec.ofInt 64 x.toInt

def inputPointer (p : Parameters) (byteBase : Nat → Nat) (b m t j : Nat) : BitVec 64 :=
  i64 (byteBase p.input.allocation+4*p.input.base)+
    signExtend64 (rowBits p.input b m)*i64 4+
    signExtend64 (colProductBits p.input 2 t j)*i64 4

def weightPointer (p : Parameters) (byteBase : Nat → Nat) (t j : Nat) : BitVec 64 :=
  i64 (byteBase p.weights.allocation+4*p.weights.base)+
    signExtend64 (colProductBits p.weights 0 t j)*i64 4

def outputPointer (p : Parameters) (byteBase : Nat → Nat) (b m t j : Nat) : BitVec 64 :=
  i64 (byteBase p.output.allocation+4*p.output.base)+
    signExtend64 (outputBits p b m t j)*i64 4

theorem i32_exact {n : Nat} (h : n < 2^31) : (i32 n).toInt = (n : Int) := by
  have hn : (i32 n).toNat = n := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]

@[simp] theorem loopBits_eq (t : Nat) : loopBits t = i32 (t*1024) :=
  BitVec.ofNat_mul_ofNat _ _

@[simp] theorem columnBits_eq (t j : Nat) : columnBits t j = i32 (t*1024+j) := by
  rw [columnBits,loopBits_eq,BitVec.ofNat_add_ofNat]

@[simp] theorem rowBits_eq (x : Tensor) (b m : Nat) :
    rowBits x b m = i32 (b*x.stride 0+m*x.stride 1) := by
  simp only [rowBits,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]

@[simp] theorem colProductBits_eq (x : Tensor) (ax t j : Nat) :
    colProductBits x ax t j = i32 ((t*1024+j)*x.stride ax) := by
  simp only [colProductBits,columnBits_eq,BitVec.ofNat_mul_ofNat]

@[simp] theorem outputBits_eq (p : Parameters) (b m t j : Nat) :
    outputBits p b m t j =
      i32 (b*p.output.stride 0+m*p.output.stride 1+(t*1024+j)*p.output.stride 2) := by
  simp only [outputBits,rowBits_eq,colProductBits_eq,BitVec.ofNat_add_ofNat]

theorem signExtend_exact {n : Nat} (h : n < 2^31) : signExtend64 (i32 n) = i64 n := by
  simp only [signExtend64,i32_exact h,BitVec.ofInt_natCast]

/-- Both loop guards and the post-last increment are represented. The bound is
exactly K≤2147482624, rather than the insufficient K<2^31. -/
theorem loop_bound_iff (p : Parameters) : trips p*1024 < 2^31 ↔ p.width ≤ 2147482624 := by
  unfold trips
  omega

theorem loop_guard (p : Parameters) (t : Nat) : t < trips p ↔ t*1024 < p.width := by
  unfold trips
  omega

theorem loop_exact (p : Parameters) (d : Domain p) (f : FiniteAdmission p d preexisting)
    (t : Nat) (ht : t < trips p) :
    (loopBits t).toInt = (t*1024 : Nat) ∧
    (loopBits t+i32 1024).toInt = ((t+1)*1024 : Nat) ∧
    ((loopBits t).slt (i32 p.width) = Bool.true ↔ t*1024 < p.width) ∧
    (loopBits (trips p)).slt (i32 p.width) = Bool.false := by
  have last := f.loopEnd
  have width := width_le_padded p
  have start : t*1024 < 2^31 := by omega
  have back : (t+1)*1024 < 2^31 := by omega
  have hw : p.width < 2^31 := by omega
  refine ⟨?_,?_,?_,?_⟩
  · rw [loopBits_eq]; exact i32_exact start
  · rw [loopBits_eq,BitVec.ofNat_add_ofNat,i32_exact (by omega)]
    norm_cast
    omega
  · rw [BitVec.slt_iff_toInt_lt,loopBits_eq,i32_exact start,i32_exact hw]
    omega
  · rw [BitVec.slt_eq_decide,loopBits_eq,i32_exact last,i32_exact hw]
    simp only [decide_eq_false_iff_not]
    omega

/-- Every dynamic stride operand, every product, and every source sum—not just
the final address—fits its observed signed i32 type. -/
def ScalarIntermediates (p : Parameters) (b m t j : Nat) : List Nat :=
  [p.width,1024,b,m,t,t*1024,(t+1)*1024,j,t*1024+j,
   p.input.stride 0,p.input.stride 1,p.input.stride 2,p.weights.stride 0,
   p.output.stride 0,p.output.stride 1,p.output.stride 2,
   b*p.input.stride 0,m*p.input.stride 1,b*p.input.stride 0+m*p.input.stride 1,
   (t*1024+j)*p.input.stride 2,(t*1024+j)*p.weights.stride 0,
   b*p.output.stride 0,m*p.output.stride 1,b*p.output.stride 0+m*p.output.stride 1,
   (t*1024+j)*p.output.stride 2,
   b*p.output.stride 0+m*p.output.stride 1+(t*1024+j)*p.output.stride 2]

theorem intermediates_exact (p : Parameters) (d : Domain p) (f : FiniteAdmission p d preexisting)
    (b m t j : Nat) (hb : b < p.batches) (hm : m < p.rows)
    (ht : t < trips p) (hj : j < 1024) :
    ∀ n ∈ ScalarIntermediates p b m t j, (i32 n).toInt = (n : Int) := by
  have padded : t*1024+j < trips p*1024 := by omega
  have hloop := f.loopEnd
  have hwidth := width_le_padded p
  have hgridX := f.gridX
  have hgridY := f.gridY
  have hi0 := f.inputStrides 0 (by decide)
  have hi1 := f.inputStrides 1 (by decide)
  have hi2 := f.inputStrides 2 (by decide)
  have hw := f.weightStride
  have ho0 := f.outputStrides 0 (by decide)
  have ho1 := f.outputStrides 1 (by decide)
  have ho2 := f.outputStrides 2 (by decide)
  have hir := f.inputRow b m hb hm
  have hic := f.inputColumn _ padded
  have hwc := f.weightColumn _ padded
  have hoc := f.outputCombined b m _ hb hm padded
  intro n member
  apply i32_exact
  simp only [ScalarIntermediates,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl <;> omega

theorem column_mask_exact (p : Parameters) (d : Domain p) (f : FiniteAdmission p d preexisting)
    (t j : Nat) (ht : t < trips p) (hj : j < 1024) :
    (columnBits t j).toInt = (t*1024+j : Nat) ∧
    ((columnBits t j).slt (i32 p.width) = Bool.true ↔ t*1024+j < p.width) := by
  have h := f.loopEnd
  have hw := width_le_padded p
  have hk : t*1024+j < 2^31 := by omega
  rw [columnBits_eq]
  refine ⟨i32_exact hk,?_⟩
  rw [BitVec.slt_iff_toInt_lt,i32_exact hk,i32_exact (by omega)]
  omega

theorem pointers_exact (p : Parameters) (d : Domain p) (f : FiniteAdmission p d preexisting)
    (b m t j : Nat) (hb : b < p.batches) (hm : m < p.rows)
    (ht : t < trips p) (hj : j < 1024) :
    (inputPointer p f.byteBase b m t j).toNat =
      f.byteBase p.input.allocation+4*inputOffset p b m (t*1024+j) ∧
    (weightPointer p f.byteBase t j).toNat =
      f.byteBase p.weights.allocation+4*weightOffset p (t*1024+j) ∧
    (outputPointer p f.byteBase b m t j).toNat =
      f.byteBase p.output.allocation+4*outputOffset p b m (t*1024+j) := by
  have padded : t*1024+j < trips p*1024 := by omega
  have ir := f.inputRow b m hb hm
  have ic := f.inputColumn _ padded
  have wc := f.weightColumn _ padded
  have oc := f.outputCombined b m _ hb hm padded
  have ip := f.inputPadded b m _ hb hm padded
  have wp := f.weightPadded _ padded
  have op := f.outputPadded b m _ hb hm padded
  refine ⟨?_,?_,?_⟩
  · simp only [inputPointer,rowBits_eq,colProductBits_eq,signExtend_exact ir,signExtend_exact ic,
      BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
    have equal : f.byteBase p.input.allocation+4*p.input.base+
        (b*p.input.stride 0+m*p.input.stride 1)*4+(t*1024+j)*p.input.stride 2*4 =
        f.byteBase p.input.allocation+4*inputOffset p b m (t*1024+j) := by
      unfold inputOffset; omega
    rw [equal,Nat.mod_eq_of_lt ip]
  · simp only [weightPointer,colProductBits_eq,signExtend_exact wc,
      BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
    have equal : f.byteBase p.weights.allocation+4*p.weights.base+(t*1024+j)*p.weights.stride 0*4 =
        f.byteBase p.weights.allocation+4*weightOffset p (t*1024+j) := by
      unfold weightOffset; omega
    rw [equal,Nat.mod_eq_of_lt wp]
  · simp only [outputPointer,outputBits_eq,signExtend_exact oc,
      BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
    have equal : f.byteBase p.output.allocation+4*p.output.base+
        (b*p.output.stride 0+m*p.output.stride 1+(t*1024+j)*p.output.stride 2)*4 =
        f.byteBase p.output.allocation+4*outputOffset p b m (t*1024+j) := by
      unfold outputOffset; omega
    rw [equal,Nat.mod_eq_of_lt op]

/-- Active accesses stay within their physical allocations and have aligned
complete four-byte cells. This also bounds each preexisting-output separation. -/
theorem active_accesses (p : Parameters) (d : Domain p) (f : FiniteAdmission p d preexisting)
    (b m k : Nat) (hb : b < p.batches) (hm : m < p.rows) (hk : k < p.width) :
    inputOffset p b m k < d.capacity p.input.allocation ∧
    weightOffset p k < d.capacity p.weights.allocation ∧
    outputOffset p b m k < d.capacity p.output.allocation ∧
    (∀ t ∈ [p.input,p.weights,p.output], ∀ offset, offset < d.capacity t.allocation →
      f.byteBase t.allocation+4*offset+4 ≤ 2^64 ∧ 4 ∣ f.byteBase t.allocation+4*offset) ∧
    (∀ a byte lane, preexisting a → byte < 4*d.capacity a → lane < 4 →
      f.byteBase p.output.allocation+4*outputOffset p b m k+lane ≠ f.byteBase a+byte) := by
  refine ⟨d.inputBounds b m k hb hm hk,d.weightBounds k hk,d.outputBounds b m k hb hm hk,?_,?_⟩
  · intro t member offset bound
    have alloc := f.allocationEnd t member
    exact ⟨by omega,dvd_add (f.aligned t member) (dvd_mul_right 4 offset)⟩
  · intro a byte lane existed hbyte hlane
    have bound := d.outputBounds b m k hb hm hk
    rcases f.preexistingSeparated a existed with before | after <;> omega

/-- A kernel-checked overflow witness for the unsupported large-K domain;
it is a source-machine defect and is never a deferred numerical law. -/
theorem large_width_overflow :
    (i32 2147482624+i32 1024).toInt = -2147483648 ∧
    (i32 2147482624+i32 1024).slt (i32 2147482625) = Bool.true ∧
    (i32 2147482624+i32 1024).slt (i32 2147483647) = Bool.true := by
  decide +kernel

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
