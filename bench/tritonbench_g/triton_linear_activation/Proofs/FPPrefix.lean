import bench.tritonbench_g.triton_linear_activation.Proofs.FPReference
namespace VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

structure Offsets (p : Parameters) (s t : State Word) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  offm : t.regs .nat [p.blockRows] "m_offs"=some (fun i => row p s i.1)
  offn : t.regs .nat [p.blockColumns] "n_offs"=some (fun i => col p s i.1)
  offk : t.regs .nat [p.blockInner] "k_range_offs"=some (fun i => i.1.val)
def aPointer (p : Parameters) (s : State Word) (cur : Nat) : Values Word .ptr [p.blockRows,p.blockInner] :=
  fun i => (p.input,aOffset p s cur i)
def bPointer (p : Parameters) (s : State Word) (cur : Nat) : Values Word .ptr [p.blockInner,p.blockColumns] :=
  fun i => (p.weight,bOffset p s cur i)
structure Ready (p : Parameters) (s t : State Word) (cur : Nat) (acc : Accumulator p) extends Offsets p s t : Prop where
  aptr : t.regs .ptr [p.blockRows,p.blockInner] "A"=some (aPointer p s cur)
  bptr : t.regs .ptr [p.blockInner,p.blockColumns] "B"=some (bPointer p s cur)
  acc : t.regs .real [p.blockRows,p.blockColumns] "acc"=some acc

theorem scalars_run [Inhabited Word] (M : Algebra Word) (p : Parameters) (s : State Word) :
    ∃ t, run M (scalars p) s=some t ∧ Offsets p s t := by
  simp [scalars,source,typed,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg]
  constructor <;> simp [State.setReg,row,col,gridRows,gridCols,width,groupIdx,groupSize,blockRow,blockCol]
  all_goals rfl

theorem seeds_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (tanh32 erf32 : Word → Word) (divide : Word → Word → Word) (g : Graph) (fmt saved : Format) (p : Parameters) (s t : State Word)
    (h : Offsets p s t) :
    ∃ u, run (backend rest cfg D F tanh32 erf32 divide) (seeds g fmt saved p) t=some u ∧
      Ready p s u 0 (initial cfg g fmt p s) := by
  cases g <;> cases fmt <;> cases hb : p.hasBias <;>
    simp [seeds,source,typed,fusedTyped,dtype,LoraExpandGemv.FPValue.outputType,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,Region.cast,ofFloat,toFloat,State.setReg,
      h.mem,h.pids,h.undef,h.offm,h.offn,h.offk,hb,backend,
      SgmvExpandSlice.FPValue.backend,LoraExpandGemv.FPValue.backend,LoraExpandGemv.FPValue.formatOf]
  all_goals constructor
  all_goals first
    | (constructor <;> simp [State.setReg,h.mem,h.pids,h.undef,h.offm,h.offn,h.offk])
    | (simp [State.setReg,aPointer,bPointer,aOffset,bOffset,initial,biasValue,readWord,hb,
        h.mem,h.pids,h.undef,h.offm,h.offn,h.offk,ofFloat,toFloat,castWord])
  all_goals first | rfl | (funext i; simp [aPointer,bPointer,aOffset,bOffset,bop,initial,biasValue,
    readWord,hb,h.mem,h.offn,h.offm,ofFloat,toFloat,Region.cast,castWord])
#axiomsClean scalars_run
#axiomsClean seeds_run
end
end VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
