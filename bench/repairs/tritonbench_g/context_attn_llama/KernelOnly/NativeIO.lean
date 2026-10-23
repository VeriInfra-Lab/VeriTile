import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.NativeArrays

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
variable {α : Type} [Inhabited α]

/-- Exactly the existing signed-metadata FP32 source and per-launch IDs. -/
def invocation (c : Host.Call) (M : Algebra α) (idx : GridIndex c.grid) : Invocation α where
  kernel:=kernel c.layout c.regions .fp32 (Host.scale c.layout.D)
  backend:=M.withDefaultPrecision .fp32
  pids:=idx.toPids
  numPids:=some c.grid.toNumPids

abbrev outputs (c : Host.Call) (order : List (GridIndex c.grid)) : List IOChannel :=
  [⟨.real,[order.length,c.layout.BM,c.layout.D],c.regions.out⟩]

/-- All supplied jobs, including repetitions and empty orders. Simultaneously
bound input roles may alias; only three metadata/output separations are needed
by the theorem. The output initial-value view is not a source read footprint. -/
def programIO (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (order : List (GridIndex c.grid)) : ArrayProgramIO α α where
  inputs:=NativeArrays.ports c
  outputs:=outputs c order
  scratch:=[]
  buffers:=(NativeArrays.ports c).map (·.buf)
  reads:=fun port vals=>NativeArrays.reads c M dots order vals port
  write:=fun port=>match port with | ⟨0,_⟩=>fun _ j=>c.offset order[j.1] j.2
  outputMask:=fun port=>match port with | ⟨0,_⟩=>fun _ j=>c.active order[j.1] j.2
  scratchWrite:=fun port=>Fin.elim0 port
  scratchMask:=fun port=>Fin.elim0 port
  schedule:=fun _=>order.map (invocation c M)
  encode:=id
  observe:=fun _ _=>id

/-- The final independent ordered-scatter value at every written output lane.
No injectivity or order permutation is assumed. -/
def formula (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (order : List (GridIndex c.grid)) (values : NativeArrays.Inputs c α) :
    ChannelValues α (outputs c order) :=
  fun port=>match port with | ⟨0,_⟩=>fun j=>NativeArrays.grid c M dots order values (8 : Fin 9) (c.offset order[j.1] j.2)

def nativeIO (c : Host.Call) (order : List (GridIndex c.grid)) :
    ArrayProgramIO NativeInstructions.Expr NativeInstructions.Expr :=
  programIO c (NativeInstructions.backend c) (NativeInstructions.support c) order

/-- Every replica/consumer is observed, as in the original raw theorem. -/
def observer (c : Host.Call) (order : List (GridIndex c.grid)) (ops : NativeInstructions.ScalarOps) :
    (port : Fin (outputs c order).length) → TileIndex (outputs c order)[port].shape →
      NativeInstructions.Expr → Nat → NativeInstructions.Word :=
  fun _ _ expression consumer=>
    NativeInstructions.evaluate ops (NativeInstructions.plan c.layout.BM c.layout.D) expression consumer

def nativeFormula (c : Host.Call) (order : List (GridIndex c.grid))
    (ops : NativeInstructions.ScalarOps) (values : NativeArrays.Inputs c NativeInstructions.Expr) :
    ChannelValues (Nat → NativeInstructions.Word) (outputs c order) :=
  fun port=>match port with
  | ⟨0,_⟩=>fun j=>observer c order ops (0 : Fin 1) j
      (formula c (NativeInstructions.backend c) (NativeInstructions.support c) order values (0 : Fin 1) j)

#specNonCircular formula avoiding [FP.Structural.exec,FP.Structural.run,FPSuccess.runGrid,FPReference.grid]
#specNonCircular nativeFormula avoiding [FP.Structural.exec,FP.Structural.run,FPSuccess.runGrid,FPReference.grid]
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeInterface
