import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedCumFlat

/-! Actual repaired cumulative sources: flattened PID decoding, in-place
FP32 storage, valid prefixes, full memory frame and exhausted batch guard. -/
namespace VeriTile.Tests.GatedCumulativeReal
open VeriTile.Triton VeriTile.Bench.TritonBenchG VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealCumProgram RealCumExecution
open VeriTile.Bench.Repairs.ChunkGatedAttention
open RebasedCumExecution
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
open scoped BigOperators

private abbrev p : Layout := ⟨32,4,1,8,4,4,2⟩
private def u : BlockState where
  mem := fun _ off => if off = 49 then MemCell.of .nat 77 else MemCell.real (off : ℝ)
  regs := fun _ _ _ => none
  pids := fun _ => 7
  undef := fun _ _ => 43
private def A : FlatAlloc where
  flat := "flat"
  regions := ["x"]
  base := fun _ => 100
  extent := fun r => if r = "x" then 64 else 0

private theorem disjoint : A.Disjoint := by
  intro r hr s hs hne
  simp [A] at hr hs
  exact (hne (hr.trans hs.symm)).elim
private theorem covered : ∀ r, r ∉ A.regions → A.extent r = 0 := by
  intro r h
  simpa [A] using h
private theorem collision_free : NoCollision (q p 0 u) RebasedCumAccess.localLaunch 0 := by
  rintro ⟨i,j,⟨⟩⟩ ⟨k,e,⟨⟩⟩ hi hk he
  have hi' := i.isLt
  have hk' := k.isLt
  have hj := j.isLt
  have he' := e.isLt
  dsimp [offset,ChunkCumsumVector.RealStep.address,p,q,RebasedCumAccess.tileLayout,RebasedCumAccess.tileBase,RebasedCumAccess.localLaunch,decoded,RebasedCumExecution.chunk,linear,u] at he
  have hij : i = k ∧ j = e := ⟨Fin.ext (by omega),Fin.ext (by omega)⟩
  rcases hij with ⟨rfl,rfl⟩
  rfl
private theorem bounds : RebasedCumSafety.Bounds p (decoded p 0 u) (RebasedCumExecution.chunk p 0 u) "x" "x" A.extent := by
  constructor <;> rintro ⟨i,j,⟨⟩⟩ hi
  all_goals
    dsimp [tileActive,ChunkCumsumVector.RealStep.active,p,q,RebasedCumAccess.tileLayout,RebasedCumAccess.tileBase,RebasedCumAccess.localLaunch,decoded,RebasedCumExecution.chunk,linear,u] at hi
    norm_num [offset,ChunkCumsumVector.RealStep.address,p,q,RebasedCumAccess.tileLayout,RebasedCumAccess.tileBase,RebasedCumAccess.localLaunch,decoded,RebasedCumExecution.chunk,linear,u,A]
    omega
private def first : TileIndex [p.BT,p.BS] := (0,0,PUnit.unit)
private def second : TileIndex [p.BT,p.BS] := (1,0,PUnit.unit)
private theorem first_value : expected (q p 0 u) RebasedCumAccess.localLaunch 0 (u.readMem "x") first = 50 := by
  change (∑ k ∈ (Finset.univ : Finset (Fin 4)).filter (fun k => k.val ≤ 0),
    u.readMem "x" (50+k.val*4))=50
  rw [Finset.sum_filter,Fin.sum_univ_succ,Fin.sum_univ_succ,Fin.sum_univ_succ]
  norm_num [Fin.sum_univ_one,u,BlockState.readMem]

private theorem second_value : expected (q p 0 u) RebasedCumAccess.localLaunch 0 (u.readMem "x") second = 104 := by
  change (∑ k ∈ (Finset.univ : Finset (Fin 4)).filter (fun k => k.val ≤ 1),
    u.readMem "x" (50+k.val*4))=104
  rw [Finset.sum_filter,Fin.sum_univ_succ,Fin.sum_univ_succ,Fin.sum_univ_succ]
  norm_num [Fin.sum_univ_one,u,BlockState.readMem]

private theorem inactive_neighbor : ¬ RebasedCumFlat.footprint A p 2 0 "x" u (A.flat,149) := by
  rintro ⟨_,rg,off,⟨_,_,⟨i,j,⟨⟩⟩,hi,he⟩,addr⟩
  dsimp [A,FlatAlloc.addr] at addr
  dsimp [offset,ChunkCumsumVector.RealStep.address,p,q,RebasedCumAccess.tileLayout,RebasedCumAccess.tileBase,
    RebasedCumAccess.localLaunch,decoded,RebasedCumExecution.chunk,linear,u] at he
  have hj:=j.isLt
  omega

theorem both_complete_flat (swapped : Bool) : ∃ v : BlockState,
    exec (A.flattenKernel (RebasedCumProgram.source p "x" "x" 2 0 swapped .fp32 .fp32).eraseDType.toAlgKernel)
      (A.flattenState u) = some v ∧
    v.readMem A.flat 150 = 50 ∧ v.readMem A.flat 154 = 104 ∧
    v.mem A.flat 149 = MemCell.of .nat 77 := by
  obtain ⟨v,he,ho,hf⟩ := RebasedCumFlat.source_executes A disjoint covered p "x" "x" 2 0
    swapped .fp32 .fp32 u (fun _ => collision_free) (fun _ => bounds)
  refine ⟨v,he,(ho (by decide) first).trans first_value,(ho (by decide) second).trans second_value,?_⟩
  rw [← hf A.flat 149 inactive_neighbor]
  change (A.flattenState u).mem A.flat (A.addr "x" 49) = _
  rw [A.flattenState_mem_addr disjoint u (by simp [A]) (by decide)]
  rfl

theorem both_complete_region (swapped : Bool) : ∃ v : BlockState,
    exec (RebasedCumProgram.source p "x" "x" 2 0 swapped .fp32 .fp32).eraseDType.toAlgKernel u = some v ∧
    v.readMem "x" 50 = 50 ∧ v.readMem "x" 54 = 104 ∧ v.mem "x" 49 = MemCell.of .nat 77 := by
  obtain ⟨v,he,ho,hf⟩ := RebasedCumExecution.executes p "x" "x" 2 0 swapped .fp32 .fp32 u (fun _ => collision_free)
  refine ⟨v,he,(ho (by decide) first).trans first_value,(ho (by decide) second).trans second_value,?_⟩
  rw [← hf "x" 49]
  · rfl
  · rintro ⟨_,_,i,hi,he⟩
    apply inactive_neighbor
    refine ⟨rfl,"x",49,⟨by decide,rfl,i,hi,he⟩,rfl⟩

/-- An extra autotune program beyond the final batch performs no writes. -/
theorem exhausted_batch (swapped : Bool) : ∃ v : BlockState,
    exec (RebasedCumProgram.source p "x" "x" 2 1 swapped .fp32 .fp32).eraseDType.toAlgKernel u=some v ∧
      v.mem=u.mem := by
  have exhausted : ¬ enabled p 2 1 u := by decide
  obtain ⟨v,ran,_,frame⟩ := RebasedCumExecution.executes p "x" "x" 2 1 swapped .fp32 .fp32 u
    (fun active => (exhausted active).elim)
  refine ⟨v,ran,?_⟩
  funext r off
  symm
  apply frame r off
  rintro ⟨active,_⟩
  exact exhausted active

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.GatedCumulativeReal
