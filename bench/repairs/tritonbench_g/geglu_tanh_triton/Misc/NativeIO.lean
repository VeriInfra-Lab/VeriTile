import bench.repairs.tritonbench_g.geglu_tanh_triton.IO
import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.PublicIO
import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.Native

namespace VeriTile.Bench.Repairs.GegluTanhTriton.Reference
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term graphAlgebra)
noncomputable section

def nativeIO (instructions : Native.Instructions) (candidate backward : Bool)
    (count : Nat) (order : List Nat) : ProgramIO Native.Word (Term Native.Word) :=
  IO.program graphAlgebra candidate backward count order Term.input
    (Native.interpret (Native.compiledPorts instructions))

def native (instructions : Native.Instructions) (_candidate backward : Bool) (count : Nat) (_order : List Nat) :
    ChannelValues Native.Word (inputs count) → ChannelValues Native.Word (outputs count) :=
  formula backward count (Native.compiledReference instructions)

#specNonCircular Native.compiledReference avoiding [source,indexCommutedSource,selectedSource,Tile.scalar]
end
end VeriTile.Bench.Repairs.GegluTanhTriton.Reference

namespace VeriTile.Bench.Repairs.GegluTanhTriton.PublicIO
open VeriTile Triton FP.Structural
noncomputable section

theorem native (instructions : Native.Instructions) (candidate backward : Bool) (count : Nat) (order : List Nat)
    (legal : order.Perm (List.range (Grid.grid count))) :
    (Reference.nativeIO instructions candidate backward count order).Implements
      (Reference.native instructions candidate backward count order) :=
  implements _ candidate backward count order legal _ _ (Native.compiledReference instructions)
    (Native.compiled_lowering instructions)

#axiomsClean native
end
end VeriTile.Bench.Repairs.GegluTanhTriton.PublicIO
