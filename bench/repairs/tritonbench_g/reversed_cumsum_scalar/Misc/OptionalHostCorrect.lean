import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Kernels
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Contract
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.RealWrapper

/-! Complete checked wrapper correctness in the cast-erased Real interpretation.
The independent target is the ordinary suffix of the original logical input.
The original unpadded source is not certified by this revised contract. -/
namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.OptionalHostCorrectness
open VeriTile.Triton VeriTile.Torch
open scoped BigOperators
noncomputable section

def reference (logical : Nat → Nat → ℝ) (T row time : Nat) : MemCell :=
  .real (∑ k ∈ (Finset.range T).filter (fun k => time ≤ k), logical row k)

theorem python_checks_iff_preconditions (s : ArgumentMetadata) (dtype : Option DType) :
    checkInput s dtype = .ok () ↔ AdmittedArgument s dtype := checkInput_iff s dtype

/-- The original signed-int32 row product can wrap inside the admitted CUDA
grid range. The repaired source widens the PID before this multiplication. -/
theorem original_int32_row_product_wraps :
    ((BitVec.ofNat 32 1073741824) * (BitVec.ofNat 32 4)).toInt = 0 ∧
      (1073741824 * 4 : Nat) = 4294967296 := by decide

/-- Successful ordinary allocation/setup yields complete source execution and
the exact allocated return. No kernel-success or output premise is supplied. -/
theorem wrapper_correct (s : TensorMetadata) (dtype : Option DType)
    (contract : InputContract s dtype) (h : Call) (p : Storage.Placement h)
    (binding : TensorBinding s h p) (api : Host.API MemCell)
    (tried : List Config) (valid : ∀ c∈tried, c∈configurations)
    (selected : Config) (chosen : selected∈configurations)
    (caller : RealStorage.Heap) (logical : Nat → Nat → ℝ) (live : Nat → Prop)
    (laws : Host.PreparationLaws api h p RealStorage.read logical caller live)
    (a : Host.Allocated MemCell) (allocated : api.emptyLike caller=.ok a)
    (device : a.device=s.device) (launchDevice : api.launchDevice=s.device)
    (setups : h.B*h.H*h.T≠0 → Host.Setups api 0 (tried++[selected])) :
    ∃ result, Host.guarded s dtype api (RealWrapper.grid h p) h tried selected caller=.ok result ∧
      result.identity=a.identity ∧ result.device=s.device ∧ Host.metadata h p result ∧
      (∀ row : RealPhysicalStream.Row h, ∀ time, time<h.T →
        result.heap (Storage.address p.backing.output (row.val*h.T+time))=
          reference logical h.T row.val time) ∧
      RealWrapper.Good h p a.heap result.heap ∧
      (∀ addr, live addr → result.heap addr=caller addr) ∧
      (∀ addr, ¬Host.preparationWrites h p addr → result.heap addr=caller addr) ∧
      api.launchDevice=result.device := by
  have layout := binding.outputLayout
  have prepared : Host.prepare api Bool.true caller=.ok a := by simp [Host.prepare,allocated]
  obtain ⟨result,returned⟩ : ∃ result,
      Host.run api (RealWrapper.grid h p) h Bool.true tried selected caller=.ok result := by
    by_cases empty : h.B*h.H*h.T=0
    · exact ⟨a,Host.empty_return api _ h Bool.true tried selected caller a prepared empty⟩
    · obtain ⟨final,raw,_⟩ := RealWrapper.tuned_complete h layout p tried valid selected chosen a.heap
      exact ⟨{a with heap:=final},Host.successful_return api _ h Bool.true tried selected caller
        a prepared empty final raw (setups empty)⟩
  obtain ⟨identity,metadata,output,good,liveFrame,frame⟩ :=
    RealWrapper.returned_logical h layout p api Bool.true tried valid selected chosen
      caller logical live laws a prepared result returned
  have dev := (Host.returned_device api _ h Bool.true tried selected caller a result prepared returned).trans device
  refine ⟨result,(Host.guarded_eq s dtype contract api _ h tried selected caller).trans returned,
    identity,dev,metadata,?_,good,liveFrame,frame,launchDevice.trans dev.symm⟩
  intro row time ht
  rw [←binding.outputFlat row.val time row.isLt ht,output row time ht]
  exact congrArg MemCell.real
    (VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.suffix_filter _ _ _ (Nat.le_of_lt ht))

/-- Every row in a nonempty admitted grid has native-safe i64 control,
mask, stride offset and byte-pointer calculations for every tuning choice. -/
theorem native_programs (s : TensorMetadata) (dtype : Option DType)
    (contract : InputContract s dtype) (h : Call) (p : Storage.Placement h)
    (binding : TensorBinding s h p) (nonempty : 0<h.B*h.H*h.T)
    (c : Config) (chosen : c∈configurations) (row : Fin (h.B*h.H)) :
    NativeIndex.ProgramCertificate h c p.backing (binding.pid s dtype contract h p nonempty row) :=
  NativeIndex.certify h c chosen p.backing _

#axiomsClean python_checks_iff_preconditions
#axiomsClean original_int32_row_product_wraps
#axiomsClean wrapper_correct
#axiomsClean native_programs
#print axioms wrapper_correct
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.OptionalHostCorrectness
