import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Proofs.PublicIO

/-! Auxiliary single-source native correctness, retained with its historical
full declaration names and statements. This is not a two-program equivalence.
The public FPEquiv entry records the remaining comparison blocker. -/

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly
open VeriTile.Triton FP.Structural FPValue
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Actual full typed source schedule. Both original storage branches remain:
FP16/E4M3 operand payloads, FP32 live accumulation and final FP16/E4M3 storage.
No finite-value restriction or software replacement of the native MMA is used. -/
def nativeIO (rest : Algebra Word) (I : Native.Instructions) (g : Geometry)
    (ids : List Nat) : ProgramIO Word Word where
  inputs := [⟨PublicIO.cellType g.storage,[g.rows,g.inner],g.A⟩,
    ⟨PublicIO.cellType g.storage,[g.inner,g.columns],g.B⟩]
  outputs := [⟨PublicIO.cellType g.storage,[g.rows,g.columns],g.C⟩]
  scratch := []
  buffers := [g.A,g.B,g.C]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => lane.1.val*g.strideAM+lane.2.1.val*g.strideAK
    | ⟨1,_⟩ => fun lane => lane.1.val*g.strideBK+lane.2.1.val*g.strideBN
    | ⟨n+2,h⟩ => False.elim (by simp at h)
  inputMask := fun _ _ _ => True
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => max g.columns 1*lane.1.val+lane.2.1.val
    | ⟨n+1,h⟩ => False.elim (by simp at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => ids.map fun pid => {
    kernel := Source.typed g.request.parameters g.storage
    backend := backend (Native.algebra rest I) (Native.dotAcc I)
    pids := fun dim => if dim=0 then pid else 0
    numPids := some (fun dim => if dim=0 then programs g.request else 1) }
  encode := id
  observe := fun _ _ => id

/-- Independent fixed-instruction recurrence over supplied logical matrices,
including zero padding, live carry and the output storage conversion. -/
def nativeExpected (rest : Algebra Word) (I : Native.Instructions) (R : Native.Reference)
    (g : Geometry) (ids : List Nat) :
    (nativeIO rest I g ids).Inputs → (nativeIO rest I g ids).Outputs :=
  PublicIO.expected g (logicalReference (Native.numerics (Native.referenceInstructions R) g.storage) g.request)

/-- Complete source/native IO under the original four local instruction and
encoding correspondences. The full program order, all outputs and exterior
frame are preserved; native arithmetic is not identified with Real arithmetic. -/
theorem kernel_native_if_local (rest : Algebra Word) (I : Native.Instructions)
    (R : Native.Reference) (numeric : Native.Correspondence I R)
    (g : Geometry) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C≠g.A) (sepB : g.C≠g.B) :
    (nativeIO rest I g ids).Implements (nativeExpected rest I R g ids) := by
  have same := Native.instructions_eq I R numeric
  subst I
  exact PublicIO.implements g (Native.algebra rest (Native.referenceInstructions R))
    (Native.dotAcc (Native.referenceInstructions R))
    (Native.numerics (Native.referenceInstructions R) g.storage)
    (Native.primitives rest (Native.referenceInstructions R) g.storage) ids order sepA sepB
    (logicalReference (Native.numerics (Native.referenceInstructions R) g.storage) g.request)
    (reference_logical (Native.numerics (Native.referenceInstructions R) g.storage) g.request)

#specNonCircular nativeExpected avoiding [Source.typed,runPrograms,FP.Structural.exec,FP.Structural.runSchedule]
#axiomsClean kernel_native_if_local
#stmtSurfaceSubset kernel_native_if_local ⊆ [nativeIO,nativeExpected,ProgramIO.Implements,
  FP.Structural.Algebra,Word,Native.Instructions,Native.Reference,Native.Correspondence,
  RegionName,Geometry,Geometry.request,Geometry.C,Geometry.A,Geometry.B,programs,
  ProgramIO.Inputs,ProgramIO.Outputs,instInhabitedWord]
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly
