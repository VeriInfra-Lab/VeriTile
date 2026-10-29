import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Native
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.PublicFacts
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Words

namespace VeriTile.Bench.Repairs.L2NormTriton1.Native
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Words (Byte signNegation)
noncomputable section
open Classical

def adapter (p : Instructions) (N : Nat) : Source.Adapter Word Node :=
  ⟨algebra p (vectorWidth N (Host.blockSize N)),atom,fun i node => node.value (i%128),fun _ _ => rfl⟩

def callNode (call : Host.Call Word) : Host.Call Node :=
  ⟨call.input,atom call.eps,call.device,call.selectedFP32⟩

def callAdapter (p : Instructions) (call : Host.Call Word) : Source.Adapter Word Node :=
  adapter p (call.input.shape.getLastD 0)

/-- Independent native schedule at caller-entry coordinates. Epsilon is the
actual raw FP32 scalar argument, including NaNs, infinities and signed zero. -/
def reference {M : Type} (ops : Ops M Word Byte) (p : Instructions)
    (call : Host.Call Word) (entry : Host.World M) (r col : Nat) : Word :=
  row p (vectorWidth (call.input.shape.getLastD 0) (Host.blockSize (call.input.shape.getLastD 0))) (call.input.shape.getLastD 0) (Host.blockSize (call.input.shape.getLastD 0)) call.eps
    (fun j => logical ops signNegation entry.memory call.input (r*(call.input.shape.getLastD 0)+j)) col

theorem expected_native {M : Type} (ops : Ops M Word Byte) (p : Instructions)
    (call : Host.Call Word) (entry : Host.World M) (r col : Nat)
    (active : col<call.input.shape.getLastD 0) :
    Host.expected ops signNegation (callAdapter p call) (callNode call) entry r col=reference ops p call entry r col := by
  dsimp only [Host.expected,callAdapter,adapter,callNode]
  have same := Tile.scalar_congr (algebra p (vectorWidth (call.input.shape.getLastD 0) (Host.blockSize (call.input.shape.getLastD 0)))) (call.input.shape.getLastD 0)
    (Host.blockSize (call.input.shape.getLastD 0)) (atom call.eps)
    (fun j => atom (Host.entryRow ops signNegation (callNode call) (call.input.shape.getLastD 0) r entry j))
    (fun j => atom (logical ops signNegation entry.memory call.input (r*(call.input.shape.getLastD 0)+j)))
    (fun j hj => by simp only [Host.entryRow,callNode,if_pos hj]) col active
  dsimp only [callNode] at same
  rw [same,source_graph_lowering]
  rfl

#specNonCircular reference avoiding [Tile.scalar,Host.expected,Host.execute,algebra]
#axiomsClean expected_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Native
