import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Model
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Normalization
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section

/-- Whole source grid with per-program carries and actual host sum plans. -/
def ForwardCall.sourceRun (call : ForwardCall α s caller) (a : Arithmetic α) (p : PlanBinding s) : ForwardReturn α :=
  {call.run a p with memory := (sourceForward a s p.plans call.args.buffers call.allocation
    call.args.seed.isSome call.args.outputFinal call.before)}

def BackwardCall.sourceRun (call : BackwardCall α s ctx snapshot current caller)
    (a : Arithmetic α) (p : PlanBinding s) : BackwardReturn α :=
  {call.run a p with memory := (sourceBackward a s p.plans call.buffers call.allocation
    ctx.initial.isSome (call.final.map TensorObject.view).isSome call.before)}

theorem ForwardCall.sourceRun_eq (call : ForwardCall α s caller) (a : Arithmetic α) (p : PlanBinding s) :
    call.sourceRun a p=call.run a p := by
  simp only [sourceRun,sourceForward_normalized,ForwardCall.run,executeForward]

theorem BackwardCall.sourceRun_eq (call : BackwardCall α s ctx snapshot current caller)
    (a : Arithmetic α) (p : PlanBinding s) : call.sourceRun a p=call.run a p := by
  simp only [sourceRun,sourceBackward_normalized,BackwardCall.run,executeBackward,BackwardCall.buffers]

open Numeric

/-- Direct scalar interpretation of the compiled per-program graph; all native
ports are fixed-arity operations and no program result is an input. -/
def nativeArithmetic (ports : ScalarPorts) : Arithmetic Word where
  zero := VeriTile.Triton.FP.Value.ofBits .fp32 0
  one := VeriTile.Triton.FP.Value.ofBits .fp32 1065353216
  negFive := VeriTile.Triton.FP.Value.ofBits .fp32 3231711232
  fromNat := fun n => observeWord (ports.castU32 (BitVec.ofNat 32 n))
  add := fun x y => observeWord (ports.add x y)
  sub := fun x y => observeWord (ports.sub x y)
  mul := fun x y => observeWord (ports.mul x y)
  fma := fun x y z => observeWord (ports.fma x y z)
  exp2 := fun x => observeWord (ports.exp2FTZ x)
  scale := fun n => observeWord (ports.hostScale (BitVec.ofNat 32 n))

theorem nativeArithmetic_eq (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports) :
    nativeArithmetic ports=arithmetic u := by
  have hcast := funext laws.castU32
  have hadd := funext (fun x => funext (laws.add x))
  have hsub := funext (fun x => funext (laws.sub x))
  have hmul := funext (fun x => funext (laws.mul x))
  have hfma := funext (fun x => funext (fun y => funext (laws.fma x y)))
  have hexp := funext laws.exp2FTZ
  have hscale := funext laws.hostScale
  simp only [nativeArithmetic,arithmetic]
  congr 1
  · funext n
    exact laws.castU32 _
  · funext n
    exact laws.hostScale _

def ForwardCall.nativeRun (call : ForwardCall Word s caller) (u : UnaryModels) (ports : ScalarPorts)
    (_laws : ScalarLaws u ports) (p : PlanBinding s) : ForwardReturn Word :=
  call.sourceRun (nativeArithmetic ports) p

def BackwardCall.nativeRun (call : BackwardCall Word s ctx snapshot current caller) (u : UnaryModels)
    (ports : ScalarPorts) (_laws : ScalarLaws u ports) (p : PlanBinding s) : BackwardReturn Word :=
  call.sourceRun (nativeArithmetic ports) p

theorem ForwardCall.native_eq (call : ForwardCall Word s caller) (u : UnaryModels) (ports : ScalarPorts)
    (laws : ScalarLaws u ports) (p : PlanBinding s) :
    call.nativeRun u ports laws p=call.sourceRun (arithmetic u) p := by
  simp only [nativeRun,nativeArithmetic_eq u ports laws]

theorem BackwardCall.native_eq (call : BackwardCall Word s ctx snapshot current caller)
    (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports) (p : PlanBinding s) :
    call.nativeRun u ports laws p=call.sourceRun (arithmetic u) p := by
  simp only [nativeRun,nativeArithmetic_eq u ports laws]

#axiomsClean nativeArithmetic_eq
#axiomsClean ForwardCall.sourceRun_eq
#axiomsClean BackwardCall.sourceRun_eq
#axiomsClean ForwardCall.native_eq
#axiomsClean BackwardCall.native_eq
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
