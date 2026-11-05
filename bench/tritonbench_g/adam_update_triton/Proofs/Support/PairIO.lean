import bench.tritonbench_g.adam_update_triton.IO
import bench.tritonbench_g.adam_update_triton.Proofs.NativeWords

/-! Pure complete-grid IO for the original/commuted runtime-scalar pair. -/
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.WholePair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term)
noncomputable section

inductive Choice where
  | original | commuted

def kernel (choice : Choice) (total n B : Nat) : ComputeKernel :=
  match choice with
  | .original => source total n B
  | .commuted => commutedSource total n B

def invocation (choice : Choice) (sem : Tile.Semantics α) (params : Scalars α)
    (total n B pid : Nat) : Invocation α :=
  { Interface.invocation sem params total n B pid with kernel := kernel choice total n B }

def programIO (choice : Choice) (native : NativeWords.Instructions)
    (params : Scalars Native.Word) (total n B : Nat) (pids : List Nat) :
    ProgramIO Native.Word (Term Native.Word) :=
  { Interface.memory Native.Word n n 0 with
    schedule := fun _ => pids.map (invocation choice
      (Native.semantics (NativeWords.ports native)) (Native.arguments params) total n B)
    encode := Term.input
    observe := fun _ _ => Native.interpret (NativeWords.ports native) }

end
end VeriTile.Bench.Repairs.AdamUpdateTriton.WholePair
