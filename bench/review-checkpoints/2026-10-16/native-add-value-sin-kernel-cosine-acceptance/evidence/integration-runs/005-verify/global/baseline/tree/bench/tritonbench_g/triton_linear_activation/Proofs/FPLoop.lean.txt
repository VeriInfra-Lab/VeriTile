import bench.tritonbench_g.triton_linear_activation.Proofs.FPPrefix
namespace VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 18000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

theorem body_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (tanh32 erf32 : Word → Word) (divide : Word → Word → Word) (g : Graph)
    (fmt saved : Format) (p : Parameters) (s t : State Word) (cur : Nat) (a : Accumulator p)
    (h : Ready p s t cur a) :
    ∃ u, run (backend rest cfg D F tanh32 erf32 divide) (body g fmt saved p)
      (t.setReg "k" .nat [] (fun _ => cur))=some u ∧
      Ready p s u (cur+1) (update cfg D F g fmt p s cur a) := by
  cases g <;> cases fmt <;>
    simp [body,source,typed,fusedTyped,dtype,LoraExpandGemv.FPValue.outputType,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      ComputeDType.eraseDType,numeric,bop,natLt,numericLt,integralFloorDiv,integralMod,
      Region.cast,ofFloat,toFloat,State.setReg,h.mem,h.pids,h.undef,h.offm,h.offn,h.offk,
      h.aptr,h.bptr,h.acc,backend,SgmvExpandSlice.FPValue.backend,SgmvExpandSlice.FPValue.installAcc,
      LoraExpandGemv.FPValue.backend,LoraExpandGemv.FPValue.formatOf]
  all_goals constructor
  all_goals first
    | (constructor <;> simp [State.setReg,h.mem,h.pids,h.undef,h.offm,h.offn,h.offk])
    | (simp [State.setReg,h.mem,h.pids,h.undef,h.offm,h.offn,h.offk,h.aptr,h.bptr,h.acc,
        update,aValues,bValues,readWord,dtype,LoraExpandGemv.FPValue.outputType,
        aPointer,bPointer,aOffset,bOffset,bop,ofFloat,toFloat,Region.cast,castWord])
  all_goals first | rfl | (funext i; simp [aPointer,bPointer,aOffset,bOffset,bop,Nat.add_mul,Nat.add_assoc])

theorem loop_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (tanh32 erf32 : Word → Word) (divide : Word → Word → Word) (g : Graph)
    (fmt saved : Format) (p : Parameters) (s t : State Word) (cur : Nat) (a : Accumulator p)
    (h : Ready p s t cur a) :
    ∃ u, FP.Structural.range (backend rest cfg D F tanh32 erf32 divide) "k" cur p.blocks 1
      (body g fmt saved p) t=some u ∧
      ∃ last, Ready p s u last (foldRange (update cfg D F g fmt p s) cur p.blocks 1 a) := by
  apply range_values (backend rest cfg D F tanh32 erf32 divide) "k" (body g fmt saved p)
    (update cfg D F g fmt p s) (fun n a u => Ready p s u n a)
    (fun a u => ∃ n, Ready p s u n a) cur p.blocks 1 a t
  · intro n a u hu _
    exact body_run rest cfg D F tanh32 erf32 divide g fmt saved p s u n a hu
  · intro n a u hu
    exact ⟨n,hu⟩
  · exact h
#axiomsClean body_run
#axiomsClean loop_run
end
end VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
