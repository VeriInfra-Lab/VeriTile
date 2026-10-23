import bench.repairs.tritonbench_g.llama_ff_triton.Kernels
import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.Bytes
import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.CheckedHost
namespace VeriTile.Bench.Repairs.LlamaFF
noncomputable section

theorem reference_consistent (a : StorageOps (BitVec 32) (BitVec 16)) : NumericLeaves a a := by
  constructor
  · constructor <;> intros <;> rfl
  all_goals intros; rfl

/-- Both source APIs, every returned logical [batch,sequence,n] word, the full
word/byte frame, actual result metadata and device restoration. Numeric work is
restricted to scalar fields and one fixed-K16 instruction component. -/
theorem whole_fp (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual) (raw : RawByteABI abi)
    (p : Parameters (BitVec 32)) (b : Buffers) (domain : HostDomain s p b)
    (binding : CompilerBinding (BitVec 32) (BitVec 16) s p b) (state : σ)
    (caller : Memory (BitVec 32) (BitVec 16))
    (allocationFrame : ∀ q,q.1≠b.output.storage → abi.observe state q=caller q) :
    (∀ batch seq n,abi.observe ((kernel_ff s binding).native abi state).2
      (outputAddress b (logicalCell domain batch seq n))=
      wordTarget reference p b selectedPlans selectedReductions (abi.observe state) (logicalCell domain batch seq n)) ∧
    (∀ q,(∀ x : Cell s,q≠outputAddress b x) →
      abi.observe ((kernel_ff s binding).native abi state).2 q=abi.observe state q) ∧
    (∀ q,q.1≠b.output.storage → abi.observe ((kernel_ff s binding).native abi state).2 q=caller q) ∧
    (∀ (x : Cell s) (i : Fin (raw.width (outputAddress b x).1)),
      raw.bytes ((kernel_ff s binding).native abi state).2
        ((outputAddress b x).1,raw.width (outputAddress b x).1*(outputAddress b x).2+i.val)=
        cellByte (wordTarget reference p b selectedPlans selectedReductions (abi.observe state) x) i.val) ∧
    (∀ storage offset,(∀ x : Cell s,(storage,offset/raw.width storage)≠outputAddress b x) →
      raw.bytes ((kernel_ff s binding).native abi state).2 (storage,offset)=raw.bytes state (storage,offset)) ∧
    (returned s b domain.batch domain.seq domain.device domain.extents).storage=b.output.storage ∧
    (returned s b domain.batch domain.seq domain.device domain.extents).shape=[domain.batch,domain.seq,s.N] ∧
    (returned s b domain.batch domain.seq domain.device domain.extents).dtype=DType.fp16 ∧
    (∀ (device : Nat) (body : Nat × σ → Nat × σ) before,(withDevice device body before).1=before.1) := by
  have h:=whole_host actual reference leaves abi p b domain binding state caller allocationFrame
  refine ⟨fun batch seq n => h.1 _,h.2.1,h.2.2,?_,?_,rfl,rfl,rfl,device_restored⟩
  · exact native_output_bytes actual reference leaves abi raw p b domain binding state
  · exact native_full_byte_frame actual abi raw binding state

/-- Widths of the actual source storage roles, supplied by the ordinary
byte/typed-load ABI. The checked mixed-width guard permits this one-width-per-
backing representation while retaining same-width input aliasing. -/
structure RoleWidths (x w v r : Contract.TensorInfo) (a : CheckedHost.Binding x w v r)
    (raw : RawByteABI (σ := σ) abi) : Prop where
  input : raw.width x.backing=2
  weight1 : raw.width w.backing=Contract.weightBytes w
  weight3 : raw.width v.backing=Contract.weightBytes w
  rms : raw.width r.backing=Contract.weightBytes w
  output : raw.width a.output.backing=2

def checkedResult (x w v r : Contract.TensorInfo) (a : CheckedHost.Binding x w v r)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (binding : CompilerBinding (BitVec 32) (BitVec 16) (Contract.shape x w)
      (CheckedHost.parameters x) (CheckedHost.buffers x w v r a.output.backing)) (state : σ) :=
  (a.output,((kernel_ff (Contract.shape x w) binding).native abi state).2)

/-- Complete successful native result under the explicit Python guards.
Host/index/nonoverlap facts are derived here; no caller-supplied HostDomain or
whole-output numerical equation is accepted. Every returned half has two bytes. -/
theorem checked_fp (x w v r : Contract.TensorInfo) (ok : Contract.Admitted x w v r)
    (a : CheckedHost.Binding x w v r)
    (actual reference : StorageOps (BitVec 32) (BitVec 16)) (leaves : NumericLeaves actual reference)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual) (raw : RawByteABI abi)
    (widths : RoleWidths x w v r a raw)
    (binding : CompilerBinding (BitVec 32) (BitVec 16) (Contract.shape x w)
      (CheckedHost.parameters x) (CheckedHost.buffers x w v r a.output.backing)) (state : σ) :
    let result := checkedResult x w v r a abi binding state
    let b := CheckedHost.buffers x w v r a.output.backing
    result.1=a.output ∧ result.1.shape=[Contract.batch x,Contract.sequence x,Contract.outputs w] ∧
    result.1.dtype=.other "float16" ∧ result.1.device=x.device ∧
    (∀ cell : Cell (Contract.shape x w),∀ byte : Fin 2,
      raw.bytes result.2 (a.output.backing,2*(outputAddress b cell).2+byte.val)=
        cellByte (wordTarget reference (CheckedHost.parameters x) b selectedPlans selectedReductions
          (abi.observe state) cell) byte.val) ∧
    (∀ storage offset,(∀ cell : Cell (Contract.shape x w),
        (storage,offset/raw.width storage)≠outputAddress b cell) →
      raw.bytes result.2 (storage,offset)=raw.bytes state (storage,offset)) := by
  dsimp only
  let d:=CheckedHost.domain x w v r ok a
  have h:=whole_fp actual reference leaves abi raw (CheckedHost.parameters x) _ d binding state
    (abi.observe state) (fun _ _ => rfl)
  refine ⟨rfl,a.outputShape,a.outputType,a.outputDevice,?_,h.2.2.2.2.1⟩
  intro cell byte
  have width : raw.width (outputAddress (CheckedHost.buffers x w v r a.output.backing) cell).1=2 := widths.output
  have hb:=h.2.2.2.1 cell ⟨byte.val,by rw [width];exact byte.isLt⟩
  simpa only [width,outputAddress,CheckedHost.buffers,View3.address,checkedResult,widths.output] using hb

/-- The entire caller storage, including bytes outside all logical input
windows, is preserved. The allocation premise is only an allocator frame. -/
theorem checked_caller_bytes (x w v r : Contract.TensorInfo) (ok : Contract.Admitted x w v r)
    (a : CheckedHost.Binding x w v r)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual) (raw : RawByteABI abi)
    (binding : CompilerBinding (BitVec 32) (BitVec 16) (Contract.shape x w)
      (CheckedHost.parameters x) (CheckedHost.buffers x w v r a.output.backing))
    (state : σ) (caller : Address → Byte)
    (allocated : ∀ storage offset,storage≠a.output.backing → raw.bytes state (storage,offset)=caller (storage,offset))
    (storage offset : Nat) (outside : storage≠a.output.backing) :
    raw.bytes (checkedResult x w v r a abi binding state).2 (storage,offset)=caller (storage,offset) := by
  have frame:=native_full_byte_frame actual abi raw binding state storage offset
    (fun cell equal => outside (congrArg Prod.fst equal))
  exact frame.trans (allocated storage offset outside)

#axiomsClean checked_fp
#axiomsClean checked_caller_bytes
/-- Structural source/reference equality with one arbitrary operation record;
zero new numeric assumptions does not mean native hardware conformance. -/
theorem callee_native (a : Ops F) (p : Parameters F) (x : RowInput F)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (n : Nat) :
    ff_llama a p x plans reduction n=referenceValue a p x plans reduction n := value_reference a p x plans reduction n

#axiomsClean reference_consistent
#axiomsClean whole_fp
#axiomsClean callee_native
#specNonCircular referenceValue avoiding [value,accumulation,moment,projection,run,Code.native]
#specNonCircular wordTarget avoiding [cellValue,run,gridCode,Code.native,boundExecute,hostCode]
end
end VeriTile.Bench.Repairs.LlamaFF
