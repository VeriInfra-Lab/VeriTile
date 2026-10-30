import bench.repairs.tritonbench_g.f8_conversion_utils.Contract
import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.CheckedResults

namespace VeriTile.Bench.Repairs.F8ConversionUtils.KernelContract
open VeriTile.Triton FP.Structural Checked Source.BitAlgebra
noncomputable section

/-- A concrete bit-register machine. The scalar conversion is fixed by the
independent formula; all source loads, masks, stores, and dtype tags execute
through Structural. This definition makes no hardware-conformance claim. -/
def model (policy : Numerics.Policy) (p : Source.Profile) : Algebra Scalar :=
  algebra p (conversionBits policy p)

def active (s : State Scalar) (N B : Nat) (j : Fin B) : Prop := s.pids 0*B+j.val<N

/-- Only active source cells need their declared input storage type. Every
input word is admitted, including exceptional and subnormal representations. -/
def InputWords (p : Source.Profile) (X : RegionName) (N B : Nat) (s : State Scalar)
    (words : Nat → Bytes.Word p.inputWidth) : Prop :=
  ∀ j : Fin B, active s N B j →
    s.mem X (s.pids 0*B+j.val) = Source.inputCell p (pack (words (s.pids 0*B+j.val)))

/-- Successful execution of the entire shared typed source, independent
bit formula at every active output, and all inactive/other typed cells framed.
This is the symbolic kernel theorem; the next theorem supplies the finite
native bridge for every actual checked launch, without a caller size guard. -/
theorem complete (policy : Numerics.Policy) (p : Source.Profile)
    (Y X : RegionName) (N B : Nat) (initial : State Scalar)
    (words : Nat → Bytes.Word p.inputWidth) (inputs : InputWords p X N B initial words) :
    ∃ final, FP.Structural.exec ((model policy p).withDefaultPrecision .fp32)
        (Source.kernel p Y X N B) initial = some final ∧
      (∀ j : Fin B, active initial N B j →
        unpack p.outputWidth (Source.outputValue p (final.mem Y (initial.pids 0*B+j.val))) =
          conversionBits policy p (words (initial.pids 0*B+j.val))) ∧
      (∀ r i, (r≠Y ∨ ∀ j : Fin B, active initial N B j → i≠initial.pids 0*B+j.val) →
        final.mem r i=initial.mem r i) := by
  obtain ⟨final,executed,memory⟩ := Source.executes
    ((model policy p).withDefaultPrecision .fp32) p Y X N B initial
  refine ⟨final,executed,?_,?_⟩
  · intro j hj
    rw [memory,Source.memory_readback _ _ _ _ _ _ _ j hj,inputs j hj]
    have scalar := (binding p (conversionBits policy p)).conversion (words (initial.pids 0*B+j.val))
    cases p <;> simpa [model,Source.outputValue,Source.inputCell,Source.Profile.input,
      Source.Profile.output,ofFloat,toFloat,Cell.read,binding] using scalar
  · intro r i outside
    rw [memory]
    exact Source.memory_frame _ _ _ _ _ _ _ r i outside

/-- Every checked launch has an actual source run refining its entire finite
native byte program. Chunk length, signed-i32 indices, pointer arithmetic,
raw input storage and the complete output memory are linked by reused proofs.
The global Tensor count is unrestricted. No scalar conformance hypothesis is
an argument: the primitive conversion is the defined bit-machine semantics. -/
theorem native_calls (policy : Numerics.Policy) (p : Source.Profile)
    (source : Native.Allocation n p.inputWidth) (target : Native.Allocation n p.outputWidth) :
    FPCalls p (conversionBits policy p) source target :=
  fp_calls p (conversionBits policy p) source target

private theorem scatter_frame (iw ow : Nat) (mask : Nat → Bool) (pointer : Nat → Nat)
    (values : Nat → Bytes.Word ow) (lanes : List Nat) (before : Bytes.Heap) (q : Nat)
    (outside : ∀ lane∈lanes,mask lane=Bool.true → q<pointer lane ∨ pointer lane+ow≤q) :
    Lowering.scatter (Bytes.primitives iw ow) mask pointer values lanes before q=before q := by
  induction lanes generalizing before with
  | nil => rfl
  | cons lane lanes ih =>
    simp only [Lowering.scatter]
    rw [ih _ (fun j hj => outside j (by simp [hj]))]
    cases active : mask lane
    · rfl
    · exact Bytes.byte_eq _ _ _ _ q
        ((Bytes.primitives iw ow).byte_frame before (pointer lane) (values lane) q
          (outside lane (by simp) active))

/-- The entire concrete byte program frames every byte outside its actually
active stores, including inactive bytes inside the output allocation. Both
decode passes preserve this frame. No index or numerical guard is required. -/
theorem native_program_frame (policy : Numerics.Policy) (p : Source.Profile)
    (input output count pid : Nat) (before : Bytes.Heap) (q : Nat)
    (outside : ∀ lane, lane<1024 → NativeIndex.mask count pid lane=Bool.true →
      q<NativeIndex.pointer output p.outputWidth pid lane ∨
        NativeIndex.pointer output p.outputWidth pid lane+p.outputWidth≤q) :
    Lowering.program (Bytes.primitives p.inputWidth p.outputWidth) p.twice
      (conversionBits policy p) input output count pid before q=before q := by
  unfold Lowering.program
  split
  · rw [scatter_frame _ _ _ _ _ _ _ q (fun lane hl => outside lane (List.mem_range.mp hl)),
      scatter_frame _ _ _ _ _ _ _ q (fun lane hl => outside lane (List.mem_range.mp hl))]
  · exact scatter_frame _ _ _ _ _ _ _ q (fun lane hl => outside lane (List.mem_range.mp hl))

#axiomsClean complete
#axiomsClean native_calls
#axiomsClean native_program_frame
end
end VeriTile.Bench.Repairs.F8ConversionUtils.KernelContract
