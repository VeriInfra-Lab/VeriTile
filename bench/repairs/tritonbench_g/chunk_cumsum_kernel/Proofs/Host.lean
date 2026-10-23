import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.Layout

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.Host
open Layout
noncomputable section

structure Source (c : Shape) (S α : Type) where
  input : S → Coord c → α
  output : S → Coord c → α
  launch : Config → S → Option S

def sourceLaunches {c : Shape} {S α : Type} (src : Source c S α) : List Config → S → Option S
  | [],work => some work
  | cfg::rest,work => (src.launch cfg work).bind (sourceLaunches src rest)

theorem sourceLaunches_append {c : Shape} {S α : Type} (src : Source c S α)
    (first rest : List Config) (work : S) :
    sourceLaunches src (first++rest) work=(sourceLaunches src first work).bind (sourceLaunches src rest) := by
  induction first generalizing work with
  | nil => rfl
  | cons cfg tail ih =>
    simp only [List.cons_append,sourceLaunches,Option.bind_assoc]
    congr 1
    funext next
    exact ih next

end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.Host
