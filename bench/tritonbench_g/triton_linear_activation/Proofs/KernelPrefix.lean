import bench.tritonbench_g.triton_linear_activation.Proofs.KernelValues

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelPrefix
open VeriTile.Triton FP.Structural KernelSource KernelValues
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def scalars (p : Parameters) := (source .fp32 p).surfaceBody.take 13
def seeds (mode : Mode) (p : Parameters) := ((source mode p).surfaceBody.drop 13).take 4
theorem front_split (mode : Mode) (p : Parameters) : front mode p=scalars p++seeds mode p := by rfl

structure Offsets (p : Parameters) (s t : State α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  rawm : t.regs .nat [p.blockRows] "m_offs_untagged"=some (fun i => KernelValues.row p s i.1)
  rawn : t.regs .nat [p.blockColumns] "n_offs_untagged"=some (fun i => KernelValues.col p s i.1)
  offm : t.regs .nat [p.blockRows] "m_offs"=some (fun i => KernelValues.row p s i.1%p.rows)
  offn : t.regs .nat [p.blockColumns] "n_offs"=some (fun i => KernelValues.col p s i.1%p.columns)
  offk : t.regs .nat [p.blockInner] "k_range_offs"=some (fun i => i.1.val)

def aPointer (p : Parameters) (s : State α) (cur : Nat) : Values α .ptr [p.blockRows,p.blockInner] :=
  fun i => (p.input,aOffset p s cur i)
def bPointer (p : Parameters) (s : State α) (cur : Nat) : Values α .ptr [p.blockInner,p.blockColumns] :=
  fun i => (p.weight,bOffset p s cur i)
structure Ready (p : Parameters) (s t : State α) (cur : Nat) (acc : Accumulator α p)
    extends Offsets p s t : Prop where
  aptr : t.regs .ptr [p.blockRows,p.blockInner] "A"=some (aPointer p s cur)
  bptr : t.regs .ptr [p.blockInner,p.blockColumns] "B"=some (bPointer p s cur)
  acc : t.regs .real [p.blockRows,p.blockColumns] "acc"=some acc

theorem scalars_run (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, run M (scalars p) s=some t ∧ Offsets p s t := by
  simp [scalars,source,typed,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg]
  constructor <;> simp [State.setReg,KernelValues.row,KernelValues.col,gridRows,gridCols,width,groupIdx,groupSize,blockRow,blockCol]
  all_goals rfl

theorem seeds_run (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters) (s t : State α)
    (h : Offsets p s t) :
    ∃ u, run (engine M D) (seeds mode p) t=some u ∧
      Ready p s u 0 (initial M mode p s) := by
  cases mode <;> cases hb : p.hasBias <;>
    simp [seeds,source,typed,dtype,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,Region.cast,ofFloat,toFloat,State.setReg,
      h.mem,h.pids,h.undef,h.rawm,h.rawn,h.offm,h.offn,h.offk,hb,
      engine,Algebra.withDefaultPrecision,resolvePrecision]
  all_goals constructor
  all_goals first
    | (constructor <;> simp [State.setReg,h.mem,h.pids,h.undef,h.rawm,h.rawn,h.offm,h.offn,h.offk])
    | (simp [State.setReg,aPointer,bPointer,aOffset,bOffset,initial,biasValue,KernelValues.read,zero,paddedZero,hb,
        h.mem,h.pids,h.undef,h.rawm,h.rawn,h.offm,h.offn,h.offk,ofFloat,toFloat,dtype])
  all_goals first | rfl | (funext i; simp [aPointer,bPointer,aOffset,bOffset,bop,initial,biasValue,
    KernelValues.read,zero,paddedZero,hb,h.mem,h.offn,h.offm,ofFloat,toFloat,Region.cast,dtype])

#axiomsClean scalars_run
#axiomsClean seeds_run
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelPrefix
