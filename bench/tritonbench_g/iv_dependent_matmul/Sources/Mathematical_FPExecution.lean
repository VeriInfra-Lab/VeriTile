import bench.tritonbench_g.iv_dependent_matmul.Sources.Mathematical_FPReference
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

def aPointer (p : Parameters) (s : State Word) : Values Word .ptr [p.blockRows,p.blockInner] :=
  fun i => (p.input,aBase p s i)
def bPointer (p : Parameters) (s : State Word) : Values Word .ptr [p.blockInner,p.blockColumns] :=
  fun i => (p.weight,bBase p s i)
structure Ready (p : Parameters) (s t : State Word) (a : Accumulator p) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  pm : t.regs .nat [] "pid_m"=some (fun _ => rowBlock p s)
  pn : t.regs .nat [] "pid_n"=some (fun _ => colBlock p s)
  offk : t.regs .nat [p.blockInner] "offs_k"=some (fun i => i.1.val)
  aptr : t.regs .ptr [p.blockRows,p.blockInner] "a_ptr"=some (aPointer p s)
  bptr : t.regs .ptr [p.blockInner,p.blockColumns] "b_ptr"=some (bPointer p s)
  acc : t.regs .real [p.blockRows,p.blockColumns] "accumulator"=some a

theorem prologue_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (p : Parameters) (s : State Word) :
    ∃ t, run (backend rest cfg D F) (prologue p) s=some t ∧
      Ready p s t (fun _ => FP.Value.zero .fp32) := by
  simp [prologue,source,typed,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,backend,
    SgmvExpandSlice.FPValue.backend,LoraExpandGemv.FPValue.backend]
  constructor <;> simp [State.setReg,rowBlock,colBlock,columnsGrid,aPointer,bPointer,aBase,bBase,row,col,bop]
  all_goals funext i
  all_goals simp [aPointer,bPointer,aBase,bBase,row,col,rowBlock,colBlock,columnsGrid,bop]

theorem body_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (p : Parameters) (s t : State Word) (cur : Nat) (a : Accumulator p)
    (h : Ready p s t a) :
    ∃ u, run (backend rest cfg D F) (body g p) (t.setReg "k" .nat [] (fun _ => cur))=some u ∧
      Ready p s u (update cfg D F g p s cur a) := by
  cases g <;>
    simp [body,source,typed,fusedTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,
      h.mem,h.pids,h.undef,h.pm,h.pn,h.offk,h.aptr,h.bptr,h.acc,backend,
      SgmvExpandSlice.FPValue.backend,SgmvExpandSlice.FPValue.installAcc,
      LoraExpandGemv.FPValue.backend,LoraExpandGemv.FPValue.formatOf]
  all_goals constructor
  all_goals simp [State.setReg,h.mem,h.pids,h.undef,h.pm,h.pn,h.offk,h.aptr,h.bptr,h.acc,
    update,aValues,bValues,aPointer,bPointer,aBase,bBase,aOffset,bOffset,bop,ofFloat,toFloat,Region.cast]
  all_goals try dsimp [aValues,bValues,aOffset,bOffset,aBase,bBase]
  all_goals simp [show (0.0 : ℝ)=0 by norm_num,LoraExpandGemv.FPValue.addWord,bop]
  all_goals rfl

theorem loop_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (p : Parameters) (s t : State Word) (cur : Nat) (a : Accumulator p)
    (h : Ready p s t a) :
    ∃ u, FP.Structural.range (backend rest cfg D F) "k" cur (count p) 1 (body g p) t=some u ∧
      Ready p s u (foldRange (update cfg D F g p s) cur (count p) 1 a) := by
  apply range_values (backend rest cfg D F) "k" (body g p)
    (update cfg D F g p s) (fun _ a u => Ready p s u a)
    (fun a u => Ready p s u a) cur (count p) 1 a t
  · intro n a u hu _
    exact body_run rest cfg D F g p s u n a hu
  · intro n a u hu
    exact hu
  · exact h

theorem suffix_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (p : Parameters) (s t : State Word) (a : Accumulator p) (h : Ready p s t a) :
    ∃ u, run (backend rest cfg D F) (suffix p) t=some u ∧ u.mem=(written cfg p s a).mem := by
  simp [suffix,source,typed,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    Region.cast,ofFloat,toFloat,State.setReg,store,h.mem,h.pids,h.undef,h.pm,h.pn,h.offk,h.aptr,h.bptr,h.acc,
    backend,SgmvExpandSlice.FPValue.backend,LoraExpandGemv.FPValue.backend,LoraExpandGemv.FPValue.formatOf]
  simp only [written,active,outputOffset,outputValues,decide_eq_true_eq,Bool.and_eq_true]
  apply SgmvExpandSlice.FPValue.fold_memory_congr
  rfl
#axiomsClean prologue_run
#axiomsClean body_run
#axiomsClean loop_run
#axiomsClean suffix_run
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
