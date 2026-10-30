import bench.tritonbench_g.matmul_dequantize.Proofs.ScalarAddress

namespace VeriTile.Bench.Repairs.MatmulDequantize.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open scoped VeriTile.Spec
noncomputable section
variable {A : Type} [Inhabited A]

/-- Complete original gptq stage on its existing public memory interface. -/
def gptqOriginal (M : Algebra A) (D : Candidate.First.Dot A) (p : Candidate.First.Parameters) (jobs : List Nat) :=
  ProgramIO.Implementation.original (FirstIO.program M D p jobs) (fun _ => True)

def gptqScalarAddress (M : Algebra A) (D : Candidate.First.Dot A) (p : Candidate.First.Parameters) (jobs : List Nat) :=
  ProgramIO.Implementation.ofVariant (Candidate.First.variant M D p jobs) (fun _ => True)

/-- Exact discrete source optimization; all typed floating operations, masks,
branches and their order remain in the two complete actual programs. -/
specification gptq_scalar_address_equiv (M : Algebra A) (D : Candidate.First.Dot A) (p : Candidate.First.Parameters)
    (valid : Geometry.Valid (FirstGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (FirstGrid.geometry p)))) :
    gptqOriginal M D p jobs ≡[([] : Spec.Assumptions _)] gptqScalarAddress M D p jobs :=
  ProgramIO.Implementation.of_equivalent _
    (ScalarAddress.First.equivalent M D p valid ha hb hs hz jobs schedule)

#axiomsClean gptq_scalar_address_equiv

/-- Complete original s2 stage on its existing public memory interface. -/
def s2Original (M : Algebra A) (D : Candidate.Second.Dot A) (p : Candidate.Second.Parameters) (jobs : List Nat) :=
  ProgramIO.Implementation.original (SecondIO.program M D p jobs) (fun _ => True)

def s2ScalarAddress (M : Algebra A) (D : Candidate.Second.Dot A) (p : Candidate.Second.Parameters) (jobs : List Nat) :=
  ProgramIO.Implementation.ofVariant (Candidate.Second.variant M D p jobs) (fun _ => True)

/-- Exact discrete source optimization; all typed floating operations, masks,
branches and their order remain in the two complete actual programs. -/
specification s2_scalar_address_equiv (M : Algebra A) (D : Candidate.Second.Dot A) (p : Candidate.Second.Parameters)
    (valid : Geometry.Valid (SecondGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry p)))) :
    s2Original M D p jobs ≡[([] : Spec.Assumptions _)] s2ScalarAddress M D p jobs :=
  ProgramIO.Implementation.of_equivalent _
    (ScalarAddress.Second.equivalent M D p valid ha hb hs hz jobs schedule)

#axiomsClean s2_scalar_address_equiv

/-- Complete original dequant32 stage on its existing public memory interface. -/
def dequant32Original (M : Algebra A) (p : Candidate.D32.Parameters) (jobs : List Nat) :=
  ProgramIO.Implementation.original (D32IO.program M p jobs) (fun _ => True)

def dequant32ScalarAddress (M : Algebra A) (p : Candidate.D32.Parameters) (jobs : List Nat) :=
  ProgramIO.Implementation.ofVariant (Candidate.D32.variant M p jobs) (fun _ => True)

/-- Exact discrete source optimization; all typed floating operations, masks,
branches and their order remain in the two complete actual programs. -/
specification dequant32_scalar_address_equiv (M : Algebra A) (p : Candidate.D32.Parameters)
    (valid : Geometry.Valid (DequantGrid32.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry p)))) :
    dequant32Original M p jobs ≡[([] : Spec.Assumptions _)] dequant32ScalarAddress M p jobs :=
  ProgramIO.Implementation.of_equivalent _
    (ScalarAddress.D32.equivalent M p valid hb hs hz jobs schedule)

#axiomsClean dequant32_scalar_address_equiv

/-- Complete original dequant16 stage on its existing public memory interface. -/
def dequant16Original (M : Algebra A) (p : Candidate.D16.Parameters) (jobs : List Nat) :=
  ProgramIO.Implementation.original (D16IO.program M p jobs) (fun _ => True)

def dequant16ScalarAddress (M : Algebra A) (p : Candidate.D16.Parameters) (jobs : List Nat) :=
  ProgramIO.Implementation.ofVariant (Candidate.D16.variant M p jobs) (fun _ => True)

/-- Exact discrete source optimization; all typed floating operations, masks,
branches and their order remain in the two complete actual programs. -/
specification dequant16_scalar_address_equiv (M : Algebra A) (p : Candidate.D16.Parameters)
    (valid : Geometry.Valid (DequantGrid16.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry p)))) :
    dequant16Original M p jobs ≡[([] : Spec.Assumptions _)] dequant16ScalarAddress M p jobs :=
  ProgramIO.Implementation.of_equivalent _
    (ScalarAddress.D16.equivalent M p valid hb hs hz jobs schedule)

#axiomsClean dequant16_scalar_address_equiv

end
end VeriTile.Bench.Repairs.MatmulDequantize.FPEquiv
