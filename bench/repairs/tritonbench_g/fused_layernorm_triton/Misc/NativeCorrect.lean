import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.IOResults

/-! Retained single-source native correctness. This module declares ordinary
theorems only; a two-program FP equivalence is not established here. -/

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Equivalence
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section

/-- Actual complete source grid with FP32 Welford arithmetic and BF16 Y
stores. Arbitrary input expression terms remain admitted. Observed words are
embedded injectively with .bits so this common-carrier IO states the exact
old Word observations without restricting inputs to physical leaves. -/
def nativeIO (I : Native.Instructions) (P : Native.Profile)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat) (c : Contract.Configuration) :
    ProgramIO Native.Term Native.Term where
  inputs := [⟨.real,[rows,D],X⟩,⟨.real,[D],W⟩,⟨.real,[D],Bias⟩]
  outputs := [⟨.real,[rows],Mean⟩,⟨.real,[rows],Rstd⟩,⟨.bf16,[rows,D],Y⟩]
  scratch := []
  buffers := [X,W,Bias,Y,Mean,Rstd]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => lane.2.1.val+D*lane.1.val
    | ⟨1,_⟩ => fun lane => lane.1.val
    | ⟨2,_⟩ => fun lane => lane.1.val
    | ⟨n+3,h⟩ => False.elim (by simp at h)
  inputMask := fun _ _ _ => True
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun lane => lane.1.val
    | ⟨1,_⟩ => fun lane => lane.1.val
    | ⟨2,_⟩ => fun lane => lane.2.1.val+D*lane.1.val
    | ⟨n+3,h⟩ => False.elim (by simp at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (List.range rows).map fun row => {
    kernel := Contract.source X W Bias Y Mean Rstd rows D c
    backend := withEquality Native.algebra (Native.equal I P)
    pids := fun dim => if dim=0 then row else 0 }
  encode := id
  observe := fun _ _ => (fun word => (.bits word : Native.Term)) ∘ Native.observe I P

/-- Independent ordered Welford/reference arithmetic, followed by the retained
native instruction profile and actual output conversions. -/
def nativeFormula (I : Native.Instructions) (P : Native.Profile)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat) (c : Contract.Configuration) :
    (nativeIO I P X W Bias Y Mean Rstd rows D c).Inputs →
      (nativeIO I P X W Bias Y Mean Rstd rows D c).Outputs :=
  PublicIO.expected X W Bias Y Mean Rstd rows D (fun xs ws bs row =>
    let ref := Native.reference I P (xs row) ws bs D c.block c.tree
    (.bits ref.mean,.bits ref.rstd,fun col => .bits (ref.output col)))

/-- Source-to-native observation under the retained primitive instruction and
compiler profile boundary, with all three outputs and the exact exterior frame.
This is not an optimization rewrite or equality with exact Real arithmetic. -/
theorem kernel_native (I : Native.Instructions) (P : Native.Profile)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat)
    (sep : Separation X W Bias Y Mean Rstd) (c : Contract.Configuration) :
    (nativeIO I P X W Bias Y Mean Rstd rows D c).Implements
      (nativeFormula I P X W Bias Y Mean Rstd rows D c) :=
  IOResults.native I P X W Bias Y Mean Rstd rows D sep c

#specNonCircular nativeFormula avoiding [FP.Structural.exec,FP.Structural.runSchedule,
  Whole.rowValue,Whole.rowOutput,FPStages.treeValue,Whole.laneRecurrence,Whole.runRows,Contract.source]
#axiomsClean kernel_native
#stmtSurfaceSubset kernel_native ⊆ [nativeIO,nativeFormula,ProgramIO.Implements,
  Native.Instructions,Native.Profile,Native.Term,RegionName,Separation,Contract.Configuration,
  ProgramIO.Inputs,ProgramIO.Outputs,VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.instInhabitedTerm]
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Equivalence
