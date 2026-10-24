import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Host
namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001
open VeriTile.Triton FP.Structural
def OutputPost {α : Type} (x : Host.Arguments) (initial final : State α)
    (c : Host.Tensor) (value : Host.Index x → α) : Prop :=
  c.rank=2 ∧ c.rows=x.a.rows ∧ c.columns=x.b.columns ∧ c.fp32=Bool.true ∧
  c.contiguous=Bool.true ∧ c.device=x.a.device ∧ c.cuda=Bool.true ∧ c.negative=Bool.false ∧
  c.objectId≠x.a.objectId ∧ c.objectId≠x.b.objectId ∧
  (∀ i : Host.Index x,final.mem c.region (Host.address c i)=Cell.mk .real (value i)) ∧
  (∀ r o,(r≠c.region ∨ ∀ i : Host.Index x,o≠Host.address c i) → final.mem r o=initial.mem r o)
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001
