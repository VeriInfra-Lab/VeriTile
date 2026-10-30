import bench.tritonbench_g.f8_conversion_utils.Proofs.Host
import bench.tritonbench_g.f8_conversion_utils.Proofs.RealStorage
import bench.tritonbench_g.f8_conversion_utils.Proofs.BitAlgebra
import bench.tritonbench_g.f8_conversion_utils.Proofs.Numerics

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Checked
open VeriTile.Torch VeriTile.Triton
noncomputable section
variable {d : Direction} {f : FormatArgument} {x : TensorMetadata} {identity : Nat}

def MetadataSpec (identity : Nat) (d : Direction) (x : TensorMetadata) (result : TensorResult) : Prop :=
  result.identity ≠ identity ∧ result.metadata.shape = x.shape ∧
  result.metadata.dtype = outputDType d ∧ result.metadata.device = x.device ∧
  result.metadata.numel = x.numel

theorem allocated_metadata (a : Allocated identity d f x) : MetadataSpec identity d x a.result := by
  exact ⟨a.fresh,a.sameShape,a.dtype,a.sameDevice,congrArg List.prod a.sameShape⟩

theorem real_source (p : Source.Profile) (Y X : RegionName) (N B : Nat) :
    RealSource.kernel p.direction p.encoder p.format Y X N B = Source.kernel p Y X N B := by
  cases p <;> rfl

def RealCalls (p : Source.Profile) (source : Native.Allocation n p.inputWidth)
    (target : Native.Allocation n p.outputWidth) : Prop :=
  ∀ c pid,pid<NativeIndex.grid (Chunking.chunkLength n c) → ∀ before,
    RealSource.ProgramResult p.direction p.encoder p.format source target c pid before

theorem real_calls (p : Source.Profile) (source : Native.Allocation n p.inputWidth)
    (target : Native.Allocation n p.outputWidth) : RealCalls p source target := by
  intro c pid launched before
  exact RealSource.programs p.direction p.encoder p.format source target c pid launched before

def FPCalls (p : Source.Profile) (native : Bytes.Word p.inputWidth → Bytes.Word p.outputWidth)
    (source : Native.Allocation n p.inputWidth) (target : Native.Allocation n p.outputWidth) : Prop :=
  ∀ c pid,pid<NativeIndex.grid (Chunking.chunkLength n c) → ∀ before,
    let binding := Source.BitAlgebra.binding p native
    let initial := Source.backedState (Source.BitAlgebra.algebra p native) p native binding
      (Source.BitAlgebra.outputStorage p native) before
      (source.base+Chunking.chunkStart c*p.inputWidth)
      (target.base+Chunking.chunkStart c*p.outputWidth) pid
    ∃ final,FP.Structural.exec ((Source.BitAlgebra.algebra p native).withDefaultPrecision .fp32)
        (Source.kernel p "y" "x" (Chunking.chunkLength n c) 1024) initial = some final ∧
      (∀ i,binding.storeWord (Source.outputValue p (final.mem "y" i)) =
        Bytes.read p.outputWidth (Lowering.program (Bytes.primitives p.inputWidth p.outputWidth)
          p.twice native (source.base+Chunking.chunkStart c*p.inputWidth)
          (target.base+Chunking.chunkStart c*p.outputWidth) (Chunking.chunkLength n c) pid before)
          (Memory.address target.base p.outputWidth (Chunking.chunkStart c+i))) ∧
      (∀ (r : RegionName) i,r≠("y" : RegionName) → final.mem r i=initial.mem r i)

theorem fp_calls (p : Source.Profile) (native : Bytes.Word p.inputWidth → Bytes.Word p.outputWidth)
    (source : Native.Allocation n p.inputWidth) (target : Native.Allocation n p.outputWidth) :
    FPCalls p native source target := by
  intro c pid launched before
  exact Source.program_exists (Source.BitAlgebra.algebra p native) p native
    (Source.BitAlgebra.binding p native) (Source.BitAlgebra.outputStorage p native)
    source target (Chunking.chunkStart c) (Chunking.chunkLength n c) pid
    (Chunking.chunk_small n c) launched (Chunking.chunk_active n c) before

theorem checked_real (checked : checkInputs d f (.tensor x) = .ok ())
    (a : Allocated identity d f x) (before : Cells.Heap) (savedDevice : Nat)
    (ready : Bool) (launch : x.numel = 0 ∨ ready = Bool.true) :
    InputContract d f x ∧
    ∃ result,runChecked (Cells.primitives _ _ a.source.positive a.target.positive)
        id before savedDevice (.ok a) ready = .ok result ∧
      result.tensor = a.result ∧ MetadataSpec identity d x result.tensor ∧
      result.currentDevice = savedDevice ∧
      (∀ i,i<x.numel → result.memory (Memory.address a.target.base (profile d f x).outputWidth i) =
        before (Memory.address x.dataPtr.toNat (profile d f x).inputWidth i)) ∧
      (∀ q,¬a.target.Active q → result.memory q=before q) ∧
      RealCalls (profile d f x) a.source a.target := by
  refine ⟨(checkInputs_iff d f (.tensor x)).mp checked,_,
    checked_success _ id before savedDevice checked a ready launch,rfl,allocated_metadata a,rfl,?_,?_,
    real_calls _ a.source a.target⟩
  · intro i hi
    rw [Lowering.complete]
    simpa only [a.inputPointer] using (Cells.complete (profile d f x).twice a.source a.target a.separate before).1 i hi
  · intro q outside
    rw [Lowering.complete]
    exact (Cells.complete (profile d f x).twice a.source a.target a.separate before).2 q outside

theorem checked_fp (checked : checkInputs d f (.tensor x) = .ok ())
    (a : Allocated identity d f x) (before : Bytes.Heap) (savedDevice : Nat)
    (native : Bytes.Word (profile d f x).inputWidth → Bytes.Word (profile d f x).outputWidth)
    (policy : Numerics.Policy) (localCast : Numerics.LocalCast policy (profile d f x) native)
    (ready : Bool) (launch : x.numel = 0 ∨ ready = Bool.true) :
    InputContract d f x ∧
    ∃ result,runChecked (Bytes.primitives _ _) native before savedDevice (.ok a) ready = .ok result ∧
      result.tensor = a.result ∧ MetadataSpec identity d x result.tensor ∧
      result.currentDevice = savedDevice ∧
      (∀ i,i<x.numel → Bytes.read (profile d f x).outputWidth result.memory
          (Memory.address a.target.base (profile d f x).outputWidth i) =
        Numerics.reference policy (profile d f x)
          (Bytes.read (profile d f x).inputWidth before
            (Memory.address x.dataPtr.toNat (profile d f x).inputWidth i))) ∧
      (∀ q,¬a.target.Active q → result.memory q=before q) ∧
      result.memory = Lowering.allChunks (Bytes.primitives _ _) (profile d f x).twice
        (Numerics.reference policy (profile d f x)) a.source.base a.target.base x.numel
        (launchChunks x.numel) before ∧
      FPCalls (profile d f x) native a.source a.target := by
  refine ⟨(checkInputs_iff d f (.tensor x)).mp checked,_,
    checked_success _ native before savedDevice checked a ready launch,rfl,allocated_metadata a,rfl,?_,?_,?_,
    fp_calls _ native a.source a.target⟩
  · intro i hi
    rw [Lowering.complete]
    have value := (Native.full_execution (Bytes.primitives _ _) (profile d f x).twice native
      a.source a.target a.separate before).1 i hi
    simpa only [Bytes.primitives,a.inputPointer] using value.trans (localCast _)
  · intro q outside
    rw [Lowering.complete]
    exact Bytes.byte_eq _ _ _ _ q
      ((Native.full_execution (Bytes.primitives _ _) (profile d f x).twice native
        a.source a.target a.separate before).2 q outside)
  · have equal : native = Numerics.reference policy (profile d f x) := funext localCast
    rw [equal]

/-- Each native access is bound to the same checked Tensor data pointer.
All padded masks are proved by Chunking.finite_indices; only active pointers
must lie inside storage. Slice offsets are added exactly once. -/
theorem input_pointer (a : Allocated identity d f x) (c pid lane : Nat)
    (launched : pid<NativeIndex.grid (Chunking.chunkLength x.numel c)) (inTile : lane<1024)
    (active : pid*1024+lane<Chunking.chunkLength x.numel c) :
    NativeIndex.pointer (a.source.base+Chunking.chunkStart c*(profile d f x).inputWidth)
      (profile d f x).inputWidth pid lane =
      Memory.address x.dataPtr.toNat (profile d f x).inputWidth
        (Chunking.chunkStart c+(pid*1024+lane)) := by
  rw [Native.pointer a.source (Chunking.chunkStart c) (Chunking.chunkLength x.numel c) pid lane
    (Chunking.chunk_small x.numel c) launched inTile (Chunking.chunk_active x.numel c _ active),a.inputPointer]

#axiomsClean checked_real
#axiomsClean checked_fp
#axiomsClean input_pointer

theorem accessed_alignment (checked : checkInputs d f (.tensor x) = .ok ())
    (a : Allocated identity d f x) (c pid lane : Nat)
    (launched : pid<NativeIndex.grid (Chunking.chunkLength x.numel c)) (inTile : lane<1024)
    (active : pid*1024+lane<Chunking.chunkLength x.numel c) :
    NativeIndex.pointer (a.source.base+Chunking.chunkStart c*(profile d f x).inputWidth)
      (profile d f x).inputWidth pid lane % (profile d f x).inputWidth = 0 ∧
    NativeIndex.pointer (a.target.base+Chunking.chunkStart c*(profile d f x).outputWidth)
      (profile d f x).outputWidth pid lane % (profile d f x).outputWidth = 0 := by
  have inside := Chunking.chunk_active x.numel c _ active
  have contract : InputContract d f x := (checkInputs_iff d f (.tensor x)).mp checked
  have aligned : x.dataPtr.toNat % (profile d f x).inputWidth = 0 := by
    rw [profile_width]
    exact contract.alignment (by omega)
  rw [input_pointer a c pid lane launched inTile active,
    Native.pointer a.target (Chunking.chunkStart c) (Chunking.chunkLength x.numel c) pid lane
      (Chunking.chunk_small x.numel c) launched inTile inside]
  simp [Memory.address,Nat.add_mod,aligned,a.outputAligned]

#axiomsClean accessed_alignment
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Checked
