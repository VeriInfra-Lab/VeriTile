import bench.tritonbench_g.matmul_triton1.Proofs.SourceGrid
import bench.tritonbench_g.matmul_triton1.Proofs.FiniteSource
import bench.tritonbench_g.matmul_triton1.Proofs.Host
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteHost
open VeriTile.Triton FP.Structural

open Host
noncomputable section

def call {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (allocate : Allocator (DotProfile.Value α)) (x : Arguments) (s : State (DotProfile.Value α)) : Option (Tensor × State (DotProfile.Value α)) :=
  if Guard x then do
    let (c,allocated) ← allocate x s
    if Empty x then some (c,allocated)
    else (SourceRuntime.grid rest P (parameters x c) allocated).map (fun final => (c,final))
  else none

theorem correct {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (allocate : Allocator (DotProfile.Value α))
    (x : Arguments) (h : Valid x) (initial : State (DotProfile.Value α))
    (allocation : ∃ c allocated, allocate x initial=some (c,allocated) ∧ Allocated x initial allocated c)
    (canonical : ∀ c allocated, allocate x initial=some (c,allocated) →
      FiniteSource.Canonical (parameters x c))
    (ready : ∀ c allocated, allocate x initial=some (c,allocated) → ¬Empty x →
      FiniteIndex.Ready (parameters x c))
    (storage : ∀ c allocated, allocate x initial=some (c,allocated) →
      FiniteIndex.StorageBounds (parameters x c)) :
    ∃ c final, call rest P allocate x initial=some (c,final) ∧ Post (SourceRuntime.symbolicPrimitives α) x initial final c := by
  obtain ⟨c,allocated,ha,a⟩ := allocation
  have hv := geometry_valid x h initial allocated c a
  have unchanged : ∀ r col,Core.logical (SourceRuntime.symbolicPrimitives α) (parameters x c) allocated r col=
      Core.logical (SourceRuntime.symbolicPrimitives α) (parameters x c) initial r col :=
    Core.logical_preserved (SourceRuntime.symbolicPrimitives α) (parameters x c) initial allocated a.freshA a.freshB
      (fun _ _ _ => congrFun (congrFun a.memory _) _)
  by_cases he : Empty x
  · refine ⟨c,allocated,?_,a.rank,a.rows,a.columns,a.dtype,a.contiguous,a.device,a.cuda,a.resolved,
      a.freshObjectA,a.freshObjectB,?_,?_⟩
    · simp [call,h.guard,ha,he]
    · intro i
      rcases he with hm|hn
      · have := i.1.isLt; omega
      · have := i.2.isLt; omega
    · intro r o _; exact congrFun (congrFun a.memory r) o
  · obtain ⟨final,hgrid,hread,hframe⟩ := SourceRuntime.grid_correct rest P (parameters x c)
      (canonical c allocated ha) (ready c allocated ha he) hv
      a.freshA a.freshB allocated
    refine ⟨c,final,?_,a.rank,a.rows,a.columns,a.dtype,a.contiguous,a.device,a.cuda,a.resolved,
      a.freshObjectA,a.freshObjectB,?_,?_⟩
    · simp [call,h.guard,ha,he,hgrid]
    · intro i
      have hi := hread i trivial
      simpa only [unchanged] using hi
    · intro r o hf
      exact (hframe r o (hf.imp id (fun h i _ => h i))).trans (congrFun (congrFun a.memory r) o)

#axiomsClean correct
end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteHost
