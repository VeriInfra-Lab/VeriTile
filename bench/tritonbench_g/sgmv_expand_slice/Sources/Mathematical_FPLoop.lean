import bench.tritonbench_g.sgmv_expand_slice.Sources.Mathematical_FPPrefix
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

theorem body_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot)
    (fmt : Storage) (out : Output) (p : Parameters) (s t : State Word) (cur : Nat) (a : Accumulator p)
    (h : Ready p s t cur a) :
    ∃ final, run (backend rest cfg D) (body fmt out p)
      (t.setReg "kk" .nat [] (fun _ => cur))=some final ∧
      Ready p s final (cur+p.blockInner) (update cfg D fmt p s cur a) := by
  cases fmt <;>
    simp [body,source,halfTyped,bfloatTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,
      h.mem,h.pids,h.undef,h.pidm,h.pidn,h.batch,h.length,h.start,h.adapter,h.offm,h.offn,h.offk,
      h.ram,h.rbn,h.aptr,h.bptr,h.acc,backend,LoraExpandGemv.FPValue.backend,
      LoraExpandGemv.FPValue.formatOf,LoraExpandGemv.FPValue.addWord]
  all_goals constructor <;>
    simp [State.setReg,h.mem,h.pids,h.undef,h.pidm,h.pidn,h.batch,h.length,h.start,h.adapter,h.offm,h.offn,h.offk,
      h.ram,h.rbn,h.aptr,h.bptr,h.acc,update,aValues,bValues,readWord,storageType,aPointer,bPointer,
      aOffset,bOffset,bop,ofFloat,toFloat,Region.cast,Convert.cast,
      LoraExpandGemv.FPValue.castWord]
  all_goals first | rfl | (funext i; simp [aPointer,bPointer,aOffset,bOffset,bop,Nat.add_mul,Nat.add_assoc])

theorem loop_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot)
    (fmt : Storage) (out : Output) (p : Parameters) (s t : State Word) (cur : Nat) (a : Accumulator p)
    (h : Ready p s t cur a) :
    ∃ final, FP.Structural.range (backend rest cfg D) "kk" cur p.inner p.blockInner (body fmt out p) t=some final ∧
      ∃ last, Ready p s final last (foldRange (update cfg D fmt p s) cur p.inner p.blockInner a) := by
  apply range_values (backend rest cfg D) "kk" (body fmt out p) (update cfg D fmt p s)
    (fun n a u => Ready p s u n a) (fun a u => ∃ n, Ready p s u n a) cur p.inner p.blockInner a t
  · intro n a u hu _
    exact body_run rest cfg D fmt out p s u n a hu
  · intro n a u hu
    exact ⟨n,hu⟩
  · exact h
#axiomsClean body_run
#axiomsClean loop_run
end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
