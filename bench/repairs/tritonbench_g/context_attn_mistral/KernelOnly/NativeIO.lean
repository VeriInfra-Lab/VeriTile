import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.NativeArrays
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Native
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.IO

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph (Word ScalarOps)
noncomputable section
variable {α : Type} [Inhabited α]

/-- Actual source, complete per-program ABI and unchanged primitive backend. -/
def invocation (c : Host.Call) (o : FPBackend.Ops α) (idx : GridIndex c.grid) : Invocation α where
  kernel:=KernelSource.original c.layout c.regions (Host.scale c.layout.D)
  backend:=FP.ProfiledRewrite.engine o.algebra
  pids:=idx.toPids
  numPids:=some c.grid.toNumPids

/-- Ordered arbitrary jobs, with duplicates and overlapping output addresses.
The initial output read-view supports scatter preservation and in-place aliases. -/
def programIO (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid)) : ArrayProgramIO α α where
  inputs:=NativeArrays.ports c
  outputs:=[⟨.real,[jobs.length,c.layout.BM,c.layout.D],c.regions.out⟩]
  scratch:=[]
  buffers:=c.regions.out::c.inputs
  reads:=fun port a=>NativeArrays.reads c o jobs a port
  write:=fun ⟨0,_⟩ _ i=>c.offset (jobs[i.1]) i.2
  outputMask:=fun ⟨0,_⟩ _ i=>c.active (jobs[i.1]) i.2
  scratchWrite:=fun i=>Fin.elim0 i
  scratchMask:=fun i=>Fin.elim0 i
  schedule:=fun _=>jobs.map (invocation c o)
  encode:=id
  observe:=fun _ _=>id

/-- Independent online attention and source-ordered scatter on logical arrays.
Each output lane observes the final array, including later aliased overwrites. -/
def target (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid))
    (a : NativeArrays.Inputs c α) : (programIO c o jobs).Outputs :=
  fun ⟨0,_⟩ i=>NativeArrays.grid c o jobs a 5 (c.offset (jobs[i.1]) i.2)

/-- The original fixed source-shape SM90 routing, reduction trees, scalar
instructions and initializer fold. There is no new optimization candidate. -/
def nativeIO (c : Host.Call) (jobs : List (GridIndex c.grid)) :=
  programIO c (KernelNative.backend c.layout.D c.layout.window.toNat) jobs

/-- Every owner/consumer is observable, including arbitrary existing Expr nodes. -/
def observer (c : Host.Call) {jobs : List (GridIndex c.grid)} (ops : ScalarOps)
    (port : Fin (nativeIO c jobs).outputs.length)
    (_ : TileIndex ((nativeIO c jobs).outputs[port]).shape) (e : NativePhysical.Expr) : Nat → Word :=
  KernelNative.evaluate ops c.layout.D c.layout.window.toNat e

def nativeTarget (c : Host.Call) (jobs : List (GridIndex c.grid)) (ops : ScalarOps)
    (a : NativeArrays.Inputs c NativePhysical.Expr) : (nativeIO c jobs).ObservedOutputs (Nat → Word) :=
  (nativeIO c jobs).mapOutputs (observer c ops)
    (target c (KernelNative.backend c.layout.D c.layout.window.toNat) jobs a)

#specNonCircular target avoiding [FP.Structural.exec,FP.Structural.run,KernelReference.runGrid,KernelReference.grid]
#specNonCircular nativeTarget avoiding [FP.Structural.exec,FP.Structural.run,KernelReference.runGrid,KernelReference.grid]
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeInterface
