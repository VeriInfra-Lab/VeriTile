import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Native
import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.PublicFacts
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Words

namespace VeriTile.Bench.Repairs.L2NormTriton2.Native
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Words (Byte signNegation)
open L2NormTriton1.Host (World blockSize)
noncomputable section
open Classical

def adapter (p : Instructions) (N : Nat) : Source.Adapter Word Node :=
  ⟨algebra p (vectorWidth N (blockSize N)) (blockSize N),atom,
    fun i node => node.value (i%128),fun _ _ => rfl⟩

def callNode (call : Host.Call Word) : Host.Call Node :=
  ⟨⟨call.input,atom call.eps,call.device,call.selectedFP32⟩,call.gradient⟩

def callAdapter (p : Instructions) (call : Host.Call Word) : Source.Adapter Word Node :=
  adapter p call.cols

/-- Independent actual native graph at both original logical inputs. -/
def reference {M : Type} (ops : Ops M Word Byte) (p : Instructions)
    (call : Host.Call Word) (entry : World M) (r col : Nat) : Word :=
  row p (vectorWidth call.cols (blockSize call.cols)) call.cols (blockSize call.cols) call.eps
    (fun j => logical ops signNegation entry.memory call.input (r*call.cols+j))
    (fun j => logical ops signNegation entry.memory call.gradient (r*call.cols+j)) col

theorem expected_native {M : Type} (ops : Ops M Word Byte) (p : Instructions)
    (call : Host.Call Word) (entry : World M) (r col : Nat) (active : col<call.cols) :
    Host.expected ops signNegation (callAdapter p call) (callNode call) entry r col=reference ops p call entry r col := by
  dsimp only [Host.expected,callAdapter,adapter,callNode,Host.Call.cols]
  have same := Tile.scalar_congr (algebra p (vectorWidth call.cols (blockSize call.cols)) (blockSize call.cols))
    call.cols (blockSize call.cols) (atom call.eps)
    (fun j => atom (Host.entryRow ops signNegation (callNode call) call.cols r entry Bool.false j))
    (fun j => atom (Host.entryRow ops signNegation (callNode call) call.cols r entry Bool.true j))
    (fun j => atom (logical ops signNegation entry.memory call.input (r*call.cols+j)))
    (fun j => atom (logical ops signNegation entry.memory call.gradient (r*call.cols+j)))
    (fun j hj => by simp [Host.entryRow,callNode,Host.Call.inputFor,hj])
    (fun j hj => by simp [Host.entryRow,callNode,Host.Call.inputFor,hj]) col active
  dsimp only [callNode,Host.Call.cols] at same
  rw [same,source_graph_lowering]
  rfl

theorem row_agreement (a b : Instructions) (h : InstructionAgreement a b)
    (V N B : Nat) (eps : Word) (xs dys : Nat → Word) (i : Nat) :
    row a V N B eps xs dys i=row b V N B eps xs dys i := by
  rw [L2NormTriton1.Native.instructions_equal a b h]

#specNonCircular reference avoiding [Tile.scalar,Host.expected,Host.execute,algebra]
#axiomsClean expected_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Native
