import bench.repairs.tritonbench_g.matmul_kernel.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.MatmulKernel
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited KernelNative.Word := ⟨FP.Value.zero .fp32⟩

/-- Complete source/native IO on all FP32 input words. The .real channel tag
is the Structural floating carrier; the source AST and backend retain actual
FP32/TF32 accumulation, FP16 narrowing and final FP32 widening. -/
def nativeIO (instructions : KernelNative.Instructions) (p : Config) :
    ProgramIO KernelNative.Word KernelNative.Word where
  inputs := [⟨.real,[p.M,p.K],"A"⟩,⟨.real,[p.K,p.N],"B"⟩]
  outputs := [⟨.real,[p.M,p.N],"C"⟩]
  scratch := []
  buffers := ["A","B","C"]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => p.A.addr lane.1.val lane.2.1.val
    | ⟨1,_⟩ => fun lane => p.B.addr lane.1.val lane.2.1.val
    | ⟨n+2,h⟩ => False.elim (by simp at h)
  inputMask := fun _ _ _ => True
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => p.C.addr lane.1.val lane.2.1.val
    | ⟨n+1,h⟩ => False.elim (by simp at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (programs p).map fun pid => {
    kernel := KernelSource.source p
    backend := KernelSource.backend KernelNative.scalar
      (KernelScalar.dotAcc (KernelNative.dot instructions p.opcode))
    pids := fun d => if d=0 then pid.1 else pid.2
    setPids := fun d => decide (d<2) }
  encode := id
  observe := fun _ _ => id

/-- Independent source-ordered fixed-K8 native recurrence, with padded operands,
live carry and both final conversions. No state or execution appears in it. -/
def nativeFormula (instructions : KernelNative.Instructions) (p : Config) :
    (nativeIO instructions p).Inputs → (nativeIO instructions p).Outputs :=
  PublicIO.expected p (KernelNative.target instructions p)

/-- Every source result equals its independent native coordinate target.
`Instructions.mma8` is the retained fixed instruction model boundary, not a
numerical transformation assumption. No finite-word restriction is added. -/
theorem fp_equiv (instructions : KernelNative.Instructions) (p : Config) (hc : p.Valid) :
    (nativeIO instructions p).Implements (nativeFormula instructions p) :=
  PublicIO.implements p KernelNative.scalar
    (KernelScalar.dotAcc (KernelNative.dot instructions p.opcode))
    (KernelNative.target instructions p) (KernelInputs.native instructions p hc)

#specNonCircular nativeFormula avoiding [KernelSource.source, FP.Structural.exec,
  FP.Structural.runSchedule, KernelSource.launch, KernelSource.point]
#axiomsClean fp_equiv
#stmtSurfaceSubset fp_equiv ⊆
  [nativeIO, nativeFormula, ProgramIO.Implements, Config, Config.Valid, KernelNative.Instructions, KernelNative.Word, instInhabitedWord,
   ProgramIO.Inputs, ProgramIO.Outputs]
end
end VeriTile.Bench.Repairs.MatmulKernel
