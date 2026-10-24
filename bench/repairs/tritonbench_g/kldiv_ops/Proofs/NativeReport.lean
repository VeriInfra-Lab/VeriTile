import bench.repairs.tritonbench_g.kldiv_ops.Proofs.BackwardValues
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.NativeFull

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
open VeriTile.Triton.FP.Structural (Algebra)
noncomputable section

variable {Word : Type} [Inhabited Word] {codec : Codec Word} {num : Numerics Word}
  {M : Algebra Word} {api : Services Word} {compiler : Compiler}

theorem ForwardResult.native_pointers {prediction target : Tensor} {flag : Bool} {name : String} {eps : Word}
    {entry : State} {returned : Tensor} {final : State}
    (f : ForwardResult codec num M api compiler prediction target flag name eps entry returned final)
    (row col : Nat) (hr : row<f.trace.rows) (hc : col<f.trace.cols) (k : Fin 4) :
    (BitVec.ofNat 64 (f.trace.py.address (row*f.trace.cols+col)+k.val)).toNat=f.trace.py.address (row*f.trace.cols+col)+k.val ∧
    (BitVec.ofNat 64 (f.trace.pt.address (row*f.trace.cols+col)+k.val)).toNat=f.trace.pt.address (row*f.trace.cols+col)+k.val ∧
    (BitVec.ofNat 64 (f.trace.out.address (if f.trace.red=.none then row*f.trace.cols+col else row)+k.val)).toNat=
      f.trace.out.address (if f.trace.red=.none then row*f.trace.cols+col else row)+k.val :=
  f.launch.plan.active_pointers f.trace.capacity row col hr hc k

theorem BackwardResult.native_pointers {target grad dst : Tensor} {flag : Bool}
    {entry : State} {returned : Tensor} {final : State}
    (f : BackwardResult codec num M api compiler target grad dst flag entry returned final)
    (row col : Nat) (hr : row<f.trace.rows) (hc : col<f.trace.cols) (k : Fin 4) :
    (BitVec.ofNat 64 (f.trace.pt.address (row*f.trace.cols+col)+k.val)).toNat=f.trace.pt.address (row*f.trace.cols+col)+k.val ∧
    (BitVec.ofNat 64 (f.trace.work.address (row*f.trace.cols+col)+k.val)).toNat=f.trace.work.address (row*f.trace.cols+col)+k.val :=
  f.launch.plan.active_pointers f.trace.capacity row col hr hc k

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host
