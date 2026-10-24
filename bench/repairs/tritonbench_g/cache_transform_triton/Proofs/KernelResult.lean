import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.CacheTransform.KernelResult
open VeriTile.Triton FP.Structural KernelBinding
noncomputable section
open Classical
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

/-- Any interpretation of copied words respects the independent sequence
formula. The input equality is needed only for admitted source rows. -/
theorem mathematical_congr (mode : Mode) (lengths : List Nat) (capacity : Nat)
    (valid : ValidLengths mode lengths capacity) (f g : Nat → α)
    (same : ∀ row,row<capacity → f row=g row) :
    mathematical mode lengths f=mathematical mode lengths g := by
  rw [←preparedRows_math,←preparedRows_math]
  apply List.map_congr_left
  intro row member
  exact same row (preparedRows_bound mode lengths capacity valid row member)

/-- Symbolic coordinate labels are the exact-Real interpretation of this
payload-insensitive byte transport. They are not native uint8 encodings.
Every four-byte group carries its actual source-word address; execution uses the
unchanged canonical source, including all real metadata/address operations. -/
def realInitial (x : Inputs r memory) (h : InputContract r) (cosTable sinTable : Table r memory)
    (s : State ℝ) : State ℝ :=
  KernelSource.initial r
    (fun byte => (x.cos.view (cos_contiguous h)).address (byte/4))
    (fun byte => (x.sin.view (sin_contiguous h)).address (byte/4)) cosTable.read sinTable.read s

def realObserve (memory : Nat → ℝ) (bytes : Fin 4 → Nat) : ℝ :=
  memory (bytes ⟨0,by decide⟩)

def realAfter (z : Outputs r) (before : Nat → ℝ) (t : State ℝ) : Nat → ℝ :=
  realize z before
    (fun o => before ((t.mem "cos_output" o).read .nat))
    (fun o => before ((t.mem "sin_output" o).read .nat))

def nativeInitial (x : Inputs r memory) (h : InputContract r) (cosTable sinTable : Table r memory)
    (s : State Unit) : State Unit :=
  KernelSource.initial r (x.cosBytes h) (x.sinBytes h) cosTable.read sinTable.read s

def nativeObserve (bytes : Fin 4 → Nat) : VeriTile.Torch.FP32.Word :=
  assemble32 (fun b => bytes b % 256)

def nativeAfter (z : Outputs r) (before : Bytes) (t : State Unit) : Bytes :=
  realize z before
    (fun o => BitVec.ofNat 8 ((t.mem "cos_output" o).read .nat))
    (fun o => BitVec.ofNat 8 ((t.mem "sin_output" o).read .nat))

theorem real_cache (x : Cache metadata) (contiguous : metadata.isContiguous=Bool.true)
    (shape : metadata.shape=[capacity,width]) (memory : Nat → ℝ)
    (row : Nat) (hr : row<capacity) (col : Fin width) :
    observedCache (fun byte => (x.view contiguous).address (byte/4)) width .fp32 (realObserve memory) row col =
      x.real memory (row*width+col.val) := by
  have inside : row*width+col.val<metadata.numel := by
    simp only [VeriTile.Torch.TensorMetadata.numel,shape,List.prod_cons,List.prod_nil,Nat.mul_one]
    nlinarith [col.isLt]
  have quotient : (row*width+col.val)*4/4=row*width+col.val := by omega
  simp only [observedCache,Storage.bytes,realObserve,Nat.add_zero,quotient]
  unfold Cache.real
  rw [x.address contiguous _ inside]

theorem native_cache (x : Cache metadata) (contiguous : metadata.isContiguous=Bool.true)
    (noNeg : metadata.isNeg=Bool.false) (shape : metadata.shape=[capacity,width])
    (memory : Bytes) (row : Nat) (hr : row<capacity) (col : Fin width) :
    observedCache (fun o => (memory ((x.view contiguous).base+o)).toNat) width .fp32
      nativeObserve row col = x.word memory (row*width+col.val) := by
  have inside : row*width+col.val<metadata.numel := by
    simp only [VeriTile.Torch.TensorMetadata.numel,shape,List.prod_cons,List.prod_nil,Nat.mul_one]
    nlinarith [col.isLt]
  rw [←x.word_direct contiguous noNeg memory _ inside]
  unfold observedCache nativeObserve read32
  apply congrArg assemble32
  funext b
  simp only [Storage.bytes,VeriTile.Torch.FP32.DataView.address]
  have bound : (memory ((x.view contiguous).base+((row*width+col.val)*4+b.val))).toNat<256 :=
    (memory _).isLt
  rw [Nat.mod_eq_of_lt bound]
  congr 2
  omega

theorem output_word_bound (r : Request) (row : Fin r.outputRows) (col : Fin r.width) :
    row.val*r.width+col.val<r.outputRows*r.width := by nlinarith [row.isLt,col.isLt]

theorem output_byte_bound (r : Request) (row : Fin r.outputRows) (col : Fin r.width)
    (b : Fin 4) : (row.val*r.width+col.val)*4+b.val<r.count := by
  have := output_word_bound r row col
  unfold Request.count
  omega

theorem real_output_cos (z : Outputs r)
    (before : Nat → ℝ) (t : State ℝ) :
    realTensor z.cos (realAfter z before t)=
      observedTensor t "cos_output" r.outputRows r.width .fp32
        (realObserve before) := by
  apply congrArg List.ofFn
  funext row col
  unfold Cache.real
  rw [←z.cos.cache.address z.cos.contiguous _ (by rw [z.cos.size]; exact output_word_bound r row col)]
  change realize z before _ _ (z.cos.view.base+4*(row.val*r.width+col.val))=_
  rw [realize_cos _ _ _ _ _ (by
    have := output_byte_bound r row col ⟨0,by decide⟩
    simpa only [Nat.add_zero,Nat.mul_comm] using this)]
  simp only [realObserve,Storage.bytes,Nat.add_zero,Nat.mul_comm]

theorem real_output_sin (z : Outputs r)
    (before : Nat → ℝ) (t : State ℝ) :
    realTensor z.sin (realAfter z before t)=
      observedTensor t "sin_output" r.outputRows r.width .fp32
        (realObserve before) := by
  apply congrArg List.ofFn
  funext row col
  unfold Cache.real
  rw [←z.sin.cache.address z.sin.contiguous _ (by rw [z.sin.size]; exact output_word_bound r row col)]
  change realize z before _ _ (z.sin.view.base+4*(row.val*r.width+col.val))=_
  rw [realize_sin _ _ _ _ _ (by
    have := output_byte_bound r row col ⟨0,by decide⟩
    simpa only [Nat.add_zero,Nat.mul_comm] using this)]
  simp only [realObserve,Storage.bytes,Nat.add_zero,Nat.mul_comm]

theorem native_output_cos (z : Outputs r) (before : Bytes) (t : State Unit) :
    wordTensor z.cos (nativeAfter z before t)=
      observedTensor t "cos_output" r.outputRows r.width .fp32 nativeObserve := by
  apply congrArg List.ofFn
  funext row col
  rw [←z.cos.cache.word_direct z.cos.contiguous z.cos.noNeg _ _
    (by rw [z.cos.size]; exact output_word_bound r row col)]
  unfold read32 nativeObserve
  apply congrArg assemble32
  funext b
  change (realize z before _ _ (z.cos.view.base+4*(row.val*r.width+col.val)+b.val)).toNat=_
  rw [Nat.add_assoc,realize_cos _ _ _ _ _ (by
    simpa only [Nat.mul_comm] using output_byte_bound r row col b)]
  simp only [BitVec.toNat_ofNat,Storage.bytes,Nat.mul_comm]

theorem native_output_sin (z : Outputs r) (before : Bytes) (t : State Unit) :
    wordTensor z.sin (nativeAfter z before t)=
      observedTensor t "sin_output" r.outputRows r.width .fp32 nativeObserve := by
  apply congrArg List.ofFn
  funext row col
  rw [←z.sin.cache.word_direct z.sin.contiguous z.sin.noNeg _ _
    (by rw [z.sin.size]; exact output_word_bound r row col)]
  unfold read32 nativeObserve
  apply congrArg assemble32
  funext b
  change (realize z before _ _ (z.sin.view.base+4*(row.val*r.width+col.val)+b.val)).toNat=_
  rw [Nat.add_assoc,realize_sin _ _ _ _ _ (by
    simpa only [Nat.mul_comm] using output_byte_bound r row col b)]
  simp only [BitVec.toNat_ofNat,Storage.bytes,Nat.mul_comm]

#axiomsClean real_cache
#axiomsClean native_cache
#axiomsClean real_output_cos
#axiomsClean real_output_sin
#axiomsClean native_output_cos
#axiomsClean native_output_sin
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CacheTransform.KernelResult
