import bench.repairs.tritonbench_g.fused_recurrent_delta.IO
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelCompiler

/-! Native observation keeps arbitrary symbolic input Nodes. Only the output
payload is evaluated to words. Object decoding, primitive routing, reduction
trees and the runtime binary64-to-FP32 scale conversion remain explicit data. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Interface
open VeriTile.Triton FP.Structural KernelProgram KernelNativeCarrier
noncomputable section

def nativeForwardIO (platform : KernelCompiler.Platform) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform .forward s f) (host : UInt64) (p : Program .forward s) : ProgramIO Node Node :=
  {forwardIO (KernelCompiler.algebra platform binding host) s f scaleSlot p.head p.value with
    schedule := fun _ => [{job (KernelCompiler.algebra platform binding host) .forward s f scaleSlot p.head p.value with
      kernel := KernelCompiler.decodedSource platform binding}]}
def nativeBackwardIO (platform : KernelCompiler.Platform) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform .backward s f) (host : UInt64) (p : Program .backward s) : ProgramIO Node Node :=
  {backwardIO (KernelCompiler.algebra platform binding host) s f scaleSlot p.head p.value with
    schedule := fun _ => [{job (KernelCompiler.algebra platform binding host) .backward s f scaleSlot p.head p.value with
      kernel := KernelCompiler.decodedSource platform binding}]}

def observeNative (platform : KernelCompiler.Platform) (binding : KernelCompiler.Binding platform kind s f) (x : Node) : Word :=
  x.eval (KernelCompiler.isa platform binding) (KernelCompiler.routing platform binding)

def forwardObserver (platform : KernelCompiler.Platform) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform .forward s f) :
    (port : Fin (forwardOutputs s).length) → TileIndex (forwardOutputs s)[port].shape → Node → Word :=
  fun _ _ => observeNative platform binding
def backwardObserver (platform : KernelCompiler.Platform) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform .backward s f) :
    (port : Fin (backwardOutputs s).length) → TileIndex (backwardOutputs s)[port].shape → Node → Word :=
  fun _ _ => observeNative platform binding

def nativeForwardFormula (platform : KernelCompiler.Platform) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform .forward s f) (host : UInt64) (iv : Nat)
    (vals : ForwardInputs Node s) : ForwardOutputs Word s
  | ⟨0,_⟩ => fun j => observeNative platform binding
      (forwardReference (KernelCompiler.algebra platform binding host) s f scaleSlot iv vals (0 : Fin 3) j)
  | ⟨1,_⟩ => fun j => observeNative platform binding
      (forwardReference (KernelCompiler.algebra platform binding host) s f scaleSlot iv vals (1 : Fin 3) j)
  | ⟨2,_⟩ => fun j => observeNative platform binding
      (forwardReference (KernelCompiler.algebra platform binding host) s f scaleSlot iv vals (2 : Fin 3) j)
  | ⟨n+3,h⟩ => False.elim (by simp [forwardOutputs] at h)
def nativeBackwardFormula (platform : KernelCompiler.Platform) (s : Shape) (f : Flags)
    (binding : KernelCompiler.Binding platform .backward s f) (host : UInt64) (iv : Nat)
    (vals : BackwardInputs Node s) : BackwardOutputs Word s
  | ⟨0,_⟩ => fun j => observeNative platform binding
      (backwardReference (KernelCompiler.algebra platform binding host) s f scaleSlot iv vals (0 : Fin 6) j)
  | ⟨1,_⟩ => fun j => observeNative platform binding
      (backwardReference (KernelCompiler.algebra platform binding host) s f scaleSlot iv vals (1 : Fin 6) j)
  | ⟨2,_⟩ => fun j => observeNative platform binding
      (backwardReference (KernelCompiler.algebra platform binding host) s f scaleSlot iv vals (2 : Fin 6) j)
  | ⟨3,_⟩ => fun j => observeNative platform binding
      (backwardReference (KernelCompiler.algebra platform binding host) s f scaleSlot iv vals (3 : Fin 6) j)
  | ⟨4,_⟩ => fun j => observeNative platform binding
      (backwardReference (KernelCompiler.algebra platform binding host) s f scaleSlot iv vals (4 : Fin 6) j)
  | ⟨5,_⟩ => fun j => observeNative platform binding
      (backwardReference (KernelCompiler.algebra platform binding host) s f scaleSlot iv vals (5 : Fin 6) j)
  | ⟨n+6,h⟩ => False.elim (by simp [backwardOutputs] at h)

#specNonCircular nativeForwardFormula avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source,KernelReference.forwardInput]
#specNonCircular nativeBackwardFormula avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source,KernelBackwardReference.input]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Interface
