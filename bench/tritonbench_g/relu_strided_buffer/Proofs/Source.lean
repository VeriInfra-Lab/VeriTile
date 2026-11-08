import bench.tritonbench_g.relu_strided_buffer.Proofs.PastedTile
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_StructuralTile

/-! Source execution bridge. The readable DSL retains signed pointer-register
values; the established signed tile uses natural cell-index registers. Those
private registers differ, while every typed memory cell and undef observation
agree. Both conversions take place after the full signed address sum. -/
namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer
open StructuralTile SignedView SourceSignedBranch
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def parameters (a : HostPlan.Arguments) : Parameters :=
  ⟨⟨a.input.region,a.input.base,a.input.stride⟩,
   ⟨a.output.region,a.output.base,a.output.stride⟩,a.n,a.B,a.T,a.oneTile⟩

def views (c : Parameters) : SignedViews :=
  ⟨⟨c.input.region,c.input.base,c.input.stride⟩,
   ⟨c.output.region,c.output.base,c.output.stride⟩,c.n,c.B⟩

/-- Actual branch-local tile, with both scheduling branches retained in Kernels. -/
def perTile (c : Parameters) : List ComputeStmt := PastedTile.body c

def sourceFinal [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : Parameters) (T : Nat) (s : State α)
    (ht : s.regs .nat [] "tile_id" = some (fun _ => T)) : State α :=
  (PastedTile.execution M cmp hc c T s ht).val

theorem perTile_exact [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : Parameters) (T : Nat) (s : State α)
    (ht : s.regs .nat [] "tile_id" = some (fun _ => T)) :
    run M (perTile c) s = some (sourceFinal M cmp hc c T s ht) :=
  (PastedTile.execution M cmp hc c T s ht).property

theorem sourceFinal_mem [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : Parameters) (T : Nat) (s : State α)
    (ht : s.regs .nat [] "tile_id" = some (fun _ => T)) :
    (sourceFinal M cmp hc c T s ht).mem = tileOutput M cmp (views c) T s.mem := by
  rcases c with ⟨input,output,n,nLanes,tiles,branch⟩
  cases branch
  all_goals
    simp only [sourceFinal,PastedTile.execution]
    simp only [tileOutput,scatter_mem]
    simp [views,rawValues,scalarResult,scalarZero,
      View.address,View.signedAddress,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      ← Int.natCast_add,← Int.natCast_mul,-Nat.cast_add,-Nat.cast_mul]
    apply List.foldl_ext
    intro mem i _
    by_cases active : T*nLanes+i.1.val<n <;> simp [active] <;> norm_num

theorem tile_implementation [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : Parameters) :
    StructuralControl.TileImplementation M (tileOutput M cmp (views c)) (perTile c) := by
  intro T s ht
  refine ⟨sourceFinal M cmp hc c T s ht,perTile_exact M cmp hc c T s ht,
    ⟨sourceFinal_mem M cmp hc c T s ht,?_⟩,?_,?_⟩
  all_goals
    rcases c with ⟨input,output,n,nLanes,tiles,branch⟩
    cases branch
    all_goals
      simp only [sourceFinal,PastedTile.execution]
      simp [PastedTile.masked_regs,PastedTile.masked_undef,StructuralControl.flatState]

#axiomsClean perTile_exact
#axiomsClean sourceFinal_mem
#axiomsClean tile_implementation
end
end VeriTile.Bench.Repairs.ReluStridedBuffer
