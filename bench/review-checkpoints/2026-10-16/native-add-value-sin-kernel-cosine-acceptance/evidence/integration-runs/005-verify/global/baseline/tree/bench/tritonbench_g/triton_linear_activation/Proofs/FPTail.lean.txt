import bench.tritonbench_g.triton_linear_activation.Proofs.FPLoop
namespace VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 20000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

def saveCode (saved : Format) (p : Parameters) := (suffix .source .half saved p).take 1
def activationCode (g : Graph) (p : Parameters) := ((suffix g .half .half p).drop 1).take 4
def storeCode (fmt : Format) (p : Parameters) := (suffix .source fmt .half p).drop 5

theorem suffix_split (g : Graph) (fmt saved : Format) (p : Parameters) :
    suffix g fmt saved p=saveCode saved p ++ activationCode g p ++ storeCode fmt p := by
  cases g <;> rfl

structure TailReady (p : Parameters) (s t : State Word) (acc : Accumulator p) : Prop where
  offm : t.regs .nat [p.blockRows] "m_offs"=some (fun i => row p s i.1)
  offn : t.regs .nat [p.blockColumns] "n_offs"=some (fun i => col p s i.1)
  acc : t.regs .real [p.blockRows,p.blockColumns] "acc"=some acc

private theorem unmasked_memory {ι : Type} (l : List ι) (r : RegionName) (off : ι → Nat)
    (v : ι → Cell Word) (s t : State Word) (h : s.mem=t.mem) :
    (l.foldl (fun u i => u.write r (off i) (v i)) s).mem =
      (l.foldl (fun u i => u.write r (off i) (v i)) t).mem := by
  simpa only [if_true] using SgmvExpandSlice.FPValue.fold_memory_congr l r off (fun _ => True) v s t h

private theorem activation_cases (name : String) :
    name="tanh" ∨ name="gelu" ∨ name="fast_gelu" ∨ name="relu" ∨
      (name≠"tanh" ∧ name≠"gelu" ∧ name≠"fast_gelu" ∧ name≠"relu") := by
  by_cases ht : name="tanh" <;> by_cases hg : name="gelu" <;>
    by_cases hf : name="fast_gelu" <;> by_cases hr : name="relu" <;> tauto

theorem save_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (tanh32 erf32 : Word → Word) (divide : Word → Word → Word) (saved : Format)
    (p : Parameters) (s t : State Word) (a : Accumulator p) (h : TailReady p s t a)
    (hm : t.mem=s.mem) :
    ∃ u, run (backend rest cfg D F tanh32 erf32 divide) (saveCode saved p) t=some u ∧
      TailReady p s u a ∧ u.mem=(saveWritten cfg saved p s a).mem := by
  cases saved <;> cases hs : p.saveInputs <;>
    simp [saveCode,suffix,source,typed,dtype,LoraExpandGemv.FPValue.outputType,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      ComputeDType.eraseDType,numeric,bop,Region.cast,ofFloat,toFloat,State.setReg,store,
      h.offm,h.offn,h.acc,hs,backend,SgmvExpandSlice.FPValue.backend,
      LoraExpandGemv.FPValue.backend,LoraExpandGemv.FPValue.formatOf,saveWritten,outputCell]
  all_goals refine ⟨?_,?_⟩
  all_goals first
    | (constructor <;> simp [State.setReg,h.offm,h.offn,h.acc])
    | exact hm
    | (apply unmasked_memory; exact hm)

theorem activation_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (tanh32 erf32 : Word → Word) (divide : Word → Word → Word) (g : Graph)
    (p : Parameters) (s t : State Word) (a : Accumulator p) (h : TailReady p s t a) :
    ∃ u, run (backend rest cfg D F tanh32 erf32 divide) (activationCode g p) t=some u ∧
      TailReady p s u (fun i => activate cfg g tanh32 erf32 divide p.activation (a i)) ∧ u.mem=t.mem := by
  cases g <;>
    (rcases activation_cases p.activation with ht | hg | hf | hr | ⟨ht,hg,hf,hr⟩ <;>
    simp_all [activationCode,suffix,source,typed,fusedTyped,dtype,LoraExpandGemv.FPValue.outputType,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      ComputeDType.eraseDType,numeric,bop,ofFloat,toFloat,State.setReg,
      h.offm,h.offn,h.acc,backend,SgmvExpandSlice.FPValue.backend,
      LoraExpandGemv.FPValue.backend,LoraExpandGemv.FPValue.formatOf,
      activate,geluWord,fastGeluWord])
  all_goals constructor
  all_goals simp [State.setReg,h.offm,h.offn,h.acc,bop]
  all_goals rfl

theorem store_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (tanh32 erf32 : Word → Word) (divide : Word → Word → Word) (fmt : Format)
    (p : Parameters) (s t base : State Word) (a : Accumulator p) (h : TailReady p s t a)
    (hm : t.mem=base.mem) :
    ∃ u, run (backend rest cfg D F tanh32 erf32 divide) (storeCode fmt p) t=some u ∧
      u.mem=(storeResult cfg fmt p s base a).mem := by
  cases fmt <;>
    simp [storeCode,suffix,source,typed,dtype,LoraExpandGemv.FPValue.outputType,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      ComputeDType.eraseDType,numeric,bop,natLt,numericLt,Region.cast,ofFloat,toFloat,State.setReg,store,
      h.offm,h.offn,h.acc,backend,SgmvExpandSlice.FPValue.backend,
      LoraExpandGemv.FPValue.backend,LoraExpandGemv.FPValue.formatOf]
  all_goals simp only [storeResult,active,outputOffset,outputCell,decide_eq_true_eq,Bool.and_eq_true]
  all_goals apply SgmvExpandSlice.FPValue.fold_memory_congr
  all_goals exact hm
#axiomsClean suffix_split
#axiomsClean save_run
#axiomsClean activation_run
#axiomsClean store_run
end
end VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
