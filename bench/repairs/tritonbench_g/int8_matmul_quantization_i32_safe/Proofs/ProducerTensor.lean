import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.Tensor
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerTargets
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectComposedData

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ProducerTensor
open VeriTile.Triton FP.Structural Direct
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {F H : Type} [Inhabited F]

def geometry (r : RowRequest) (width : Nat) (positive : 0<width) : KernelProducerGrid.Geometry := ⟨r.G,r.K,width,positive⟩
def originalValues (pack : H → F) (r : RowRequest) (caller : Memory F H) (g k : Nat) : F :=
  (Tensor.encode pack (caller (logicalAddress r.input (g*r.K+k)))).read .real

def bind (pack : H → F) (r : RowRequest) (caller : Memory F H) (template : State F) : State F :=
  { template with mem := fun region i => if region="x" then
      Tensor.encode pack (caller (r.input.info.storage,r.input.info.offset+i)) else template.mem region i }

theorem original_reads (pack : H → F) (base : Nat → Nat) (r : RowRequest) (laws : TensorLaws base r.input)
    (caller : Memory F H) (template : State F) (width : Nat) (positive : 0<width) :
    KernelProducerTargets.Reads (geometry r width positive) (originalValues pack r caller) (bind pack r caller template) := by
  intro g k hg hk
  have address := row_input_address base r laws 0 0 g k hg hk
  have pos : 1 ≤ cols r.input.info := by change k < cols r.input.info at hk; omega
  have address' : (r.input.info.storage,r.input.info.offset+(g*r.K+k))=logicalAddress r.input (g*r.K+k) := by
    simpa [RowRequest.buffers,rowBuffers,QuantBuffers.inputAddress,RowRequest.K,
      max_eq_left pos,Nat.add_assoc] using address
  simp [bind,geometry,originalValues,address']

/-- Read back the producer's actual stored bytes and half words into the
same shared Tensor backing representation used by the GEMM endpoint. -/
def memory (unpack : F → H) (r : RowRequest) (data scales : TensorView) (prepared : Memory F H) (final : State F) : Memory F H :=
  fun p => if p.1=data.info.storage ∧ p.2<r.G*r.K then
    .inl (BitVec.ofInt 8 ((final.mem "q" p.2).read .int))
  else if p.1=scales.info.storage ∧ p.2<r.G then
    .inr (.inr (unpack ((final.mem "scales" p.2).read .fp16))) else prepared p

theorem data_readback (unpack : F → H) (r : RowRequest) (caller : Memory F H) (live : Nat → Prop)
    (alloc : RowAllocation r caller live) (final : State F) (g : Fin r.G) (k : Fin r.K) (byte : Byte)
    (stored : final.mem "q" (g.val*r.K+k.val)=.mk .int byte.toInt) :
    memory unpack r alloc.data alloc.scales alloc.memory final
      (logicalAddress alloc.data (g.val*r.K+k.val))=.inl byte := by
  have bound:=RowwiseQuantizationTriton.linear_bound r.G r.K g k g.isLt k.isLt
  simp [memory,logicalAddress,alloc.dataIndex _ bound,bound,stored]

theorem scale_readback (unpack : F → H) (r : RowRequest) (caller : Memory F H) (live : Nat → Prop)
    (alloc : RowAllocation r caller live) (final : State F) (g : Fin r.G) (word : F)
    (stored : final.mem "scales" g=.mk .fp16 word) :
    memory unpack r alloc.data alloc.scales alloc.memory final (logicalAddress alloc.scales g.val)=
      .inr (.inr (unpack word)) := by
  simp [memory,logicalAddress,alloc.scaleIndex _ g.isLt,alloc.distinct.symm,g.isLt,stored]

theorem frame (unpack : F → H) (r : RowRequest) (caller : Memory F H) (live : Nat → Prop)
    (alloc : RowAllocation r caller live) (final : State F) (p : Address) (hp : live p.1) :
    memory unpack r alloc.data alloc.scales alloc.memory final p=caller p := by
  have hd : p.1≠alloc.data.info.storage := by intro h; rw [h] at hp; exact alloc.dataFresh hp
  have hs : p.1≠alloc.scales.info.storage := by intro h; rw [h] at hp; exact alloc.scaleFresh hp
  simp only [memory,hd,hs,false_and,ite_false]
  exact alloc.frame p hd hs

local instance : Inhabited Word32 := ⟨FP.Value.zero .fp32⟩
theorem native_complete (f : KernelNativeModel.FloatingInstructions) (base : Nat → Nat) (r : RowRequest)
    (laws : TensorLaws base r.input) (caller : Memory Word32 Word16) (live : Nat → Prop)
    (alloc : RowAllocation r caller live) (template : State Word32) (width : Nat) (positive : 0<width) (covers : r.K≤width)
    (selection : Producer.Selection (geometry r width positive)) (rest : Algebra Word32)
    (order : Producer.Schedule (geometry r width positive)) :
    ∃ final, KernelProducerGrid.run (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8
      (geometry r width positive) order.order (bind KernelScalarProfile.pack16 r caller template)=some final ∧
      (∀ g : Fin r.G, ∀ k : Fin r.K,
        memory KernelScalarProfile.unpack16 r alloc.data alloc.scales alloc.memory final
          (logicalAddress alloc.data (g.val*r.K+k.val))=.inl
          (KernelProducerTargets.nativeByte f (geometry r width positive) selection
            (originalValues KernelScalarProfile.pack16 r caller g) k)) ∧
      (∀ g : Fin r.G,memory KernelScalarProfile.unpack16 r alloc.data alloc.scales alloc.memory final
        (logicalAddress alloc.scales g.val)=.inr (.inr
          (KernelProducerTargets.nativeStoredScale f (geometry r width positive) selection
            (originalValues KernelScalarProfile.pack16 r caller g)))) ∧
      (∀ p,live p.1 → memory KernelScalarProfile.unpack16 r alloc.data alloc.scales alloc.memory final p=caller p) := by
  obtain ⟨final,he,hq,hs,hf⟩ := Producer.native_complete f (geometry r width positive) selection covers rest
    (originalValues KernelScalarProfile.pack16 r caller) (bind KernelScalarProfile.pack16 r caller template)
    (original_reads KernelScalarProfile.pack16 base r laws caller template width positive) order
  refine ⟨final,he,?_,?_,frame KernelScalarProfile.unpack16 r caller live alloc final⟩
  · intro g k
    exact data_readback KernelScalarProfile.unpack16 r caller live alloc final g k _
      (hq g ⟨k,lt_of_lt_of_le k.isLt covers⟩ k.isLt)
  · intro g
    simpa only [KernelScalarProfile.half_roundtrip] using scale_readback KernelScalarProfile.unpack16 r caller live alloc final g _ (hs g)

theorem real_complete (base : Nat → Nat) (r : RowRequest) (laws : TensorLaws base r.input)
    (caller : Memory ℝ ℝ) (live : Nat → Prop) (alloc : RowAllocation r caller live) (template : State ℝ)
    (width : Nat) (positive : 0<width) (covers : r.K≤width) (selection : Producer.Selection (geometry r width positive))
    (rest : Algebra ℝ) (order : Producer.Schedule (geometry r width positive)) :
    ∃ final, KernelProducerGrid.run (KernelScalarProfile.mathematical selection.tree rest) realInteger8
      (geometry r width positive) order.order (bind id r caller template)=some final ∧
      (∀ g : Fin r.G, ∀ k : Fin r.K,memory id r alloc.data alloc.scales alloc.memory final
        (logicalAddress alloc.data (g.val*r.K+k.val))=.inl
          (KernelProducerTargets.mathematicalByte (originalValues id r caller g) r.K k)) ∧
      (∀ g : Fin r.G,memory id r alloc.data alloc.scales alloc.memory final (logicalAddress alloc.scales g.val)=
        .inr (.inr (KernelProducerTargets.mathematicalScale (originalValues id r caller g) r.K))) ∧
      (∀ p,live p.1 → memory id r alloc.data alloc.scales alloc.memory final p=caller p) := by
  obtain ⟨final,he,hq,hs,hf⟩ := Producer.mathematical_complete (geometry r width positive) selection covers rest
    (originalValues id r caller) (bind id r caller template) (original_reads id base r laws caller template width positive) order
  exact ⟨final,he,fun g k => data_readback id r caller live alloc final g k _
    (hq g ⟨k,lt_of_lt_of_le k.isLt covers⟩ k.isLt),fun g => scale_readback id r caller live alloc final g _ (hs g),
    frame id r caller live alloc final⟩

/-- The existing composed precheck derives the actual produced Tensor
request. The additional active-K guard propagates with its original shape. -/
def producedRequest (r : ComposedRequest) (safe : LaunchSafe (rows r.fpa.info) (cols r.b.info) (cols r.fpa.info))
    (caller : Memory F H) (live : Nat → Prop) (inputs : r.live live) (alloc : RowAllocation r.row caller live) : Tensor.Request where
  original := Direct.producedRequest r caller live inputs alloc
  safe := safe

#axiomsClean original_reads
#axiomsClean data_readback
#axiomsClean scale_readback
#axiomsClean frame
#axiomsClean native_complete
#axiomsClean real_complete
#specNonCircular originalValues avoiding [KernelProducerGrid.run, memory, Tensor.run]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ProducerTensor
