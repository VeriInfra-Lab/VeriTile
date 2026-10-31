import bench.tritonbench_g.chunk_gla_fwd.Proofs.OutputFlat

/-! Nonempty finite allocations and complete real execution with two K blocks,
partial time/K/V tiles and nonzero head/chunk/value IDs. The valid output is
three inter-chunk products plus one causal intra-chunk product; a neighboring
cell keeps its original non-real tag. Both source orders are covered. -/
namespace VeriTile.Tests.ChunkGLAOutputReal
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkGlaFwd RealOutput
open VeriTile.Bench.Repairs.ChunkGlaFwd
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

private def initial : BlockState where
  mem := fun r _ =>
    if r = "g" then MemCell.real 0
    else if r = "o" then MemCell.of .nat 99
    else MemCell.real 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
private def alloc : FlatAlloc where
  flat := "flat"
  regions := ["q","v","g","h","o","A"]
  base := fun r => if r = "q" then 0 else if r = "v" then 100
    else if r = "g" then 200 else if r = "h" then 300 else if r = "o" then 400 else 500
  extent := fun r => if r = "h" then 36 else if r = "A" then 12
    else if r = "q" ∨ r = "v" ∨ r = "g" ∨ r = "o" then 18 else 0

private theorem disjoint : alloc.Disjoint := by
  simp +decide [FlatAlloc.Disjoint,alloc,Region.mk.injEq]
private theorem covered : ∀ r, r ∉ alloc.regions → alloc.extent r = 0 := by
  intro r hn
  simp [alloc] at hn ⊢
  simp [hn]
private theorem injective : Function.Injective (outOffset initial 9 3 2 2) := by
  rintro ⟨i,j,⟨⟩⟩ ⟨k,l,⟨⟩⟩ he
  dsimp [outOffset,tIndex,vIndex,initial] at he
  have hi : i = k := Fin.ext (by omega)
  have hj : j = l := Fin.ext (by omega)
  subst k; subst l; rfl
private theorem bounds :
    RealOutputSafety.Bounds initial "q" "v" "g" "h" "o" "A" 9 3 9 3 18 3 3 3 3 2 2 2 alloc.extent := by
  constructor
  · intro kb _
    constructor <;> intro i j hi
    all_goals simp +decide [initial,alloc,Region.mk.injEq] at hi ⊢
    all_goals omega
  · constructor <;> intro i j hi
    all_goals simp +decide [initial,alloc,Region.mk.injEq] at hi ⊢
    all_goals omega
private theorem formula : cgfOutput initial "q" "v" "g" "h" "A"
    9 3 9 3 18 3 1 3 3 3 2 2 2 0 0 = 4 := by
  have hq (off : Nat) : initial.readMem "q" off = 1 := rfl
  have hg (off : Nat) : initial.readMem "g" off = 0 := rfl
  have hh (off : Nat) : initial.readMem "h" off = 1 := rfl
  have ha (off : Nat) : initial.readMem "A" off = 1 := rfl
  have hv (off : Nat) : initial.readMem "v" off = 1 := rfl
  simp only [cgfOutput]
  change (∑ kb : Fin 2, ∑ e : Fin 2, _) + (∑ j : Fin 2, _) = 4
  simp only [Fin.sum_univ_two,qgElem,qElem,gElem,hGuarded,hElem,aMasked,aElem,vGuarded,vElem,
    hq,hg,hh,ha,hv]
  norm_num [tIndex,kIndex,vIndex,initial]

private abbrev layout : RealOutputIO.Layout := ⟨9,3,9,3,18,3,3,3,3,2,2,2⟩
private abbrev regions : RealOutputIO.Regions := ⟨"q","v","g","h","o","A"⟩

theorem both_complete_flat (swapped : Bool) : ∃ sf,
    exec (alloc.flattenKernel (OutputFlat.program layout regions 1 swapped))
      (alloc.flattenState initial) = some sf ∧
    sf.readMem "flat" 417 = 4 ∧ sf.mem "flat" 416 = MemCell.of .nat 99 := by
  classical
  obtain ⟨sf,he,ho,hf⟩ := OutputFlat.source_executes layout regions 1 alloc disjoint covered
    swapped initial rfl (fun i j _ _ same => injective (by
      simpa [RealOutputIO.outputAddress,RealOutputIO.launch,outOffset,tIndex,vIndex,Nat.mul_one] using same)) bounds
  have value := ho (0,0,PUnit.unit) (by norm_num [RealOutputIO.active,RealOutputIO.launch,initial])
  have output_formula : RealOutputIO.formula layout 1 (RealOutputIO.launch initial)
      (LocalReal.outputInputs regions initial) (0,0,PUnit.unit) = 4 := by
    have hq (off : Nat) : initial.readMem "q" off = 1 := rfl
    have hg (off : Nat) : initial.readMem "g" off = 0 := rfl
    have hh (off : Nat) : initial.readMem "h" off = 1 := rfl
    have ha (off : Nat) : initial.readMem "A" off = 1 := rfl
    have hv (off : Nat) : initial.readMem "v" off = 1 := rfl
    simp only [RealOutputIO.formula,LocalReal.outputInputs,regions,layout,RealOutputIO.launch,
      hq,hg,hh,ha,hv]
    change (∑ kb : Fin 2,∑ e : Fin 2, _) + (∑ j : Fin 2,_) = 4
    norm_num [Fin.sum_univ_two,initial]

  refine ⟨sf,he,value.trans output_formula,?_⟩
  rw [← hf "flat" 416]
  · change (alloc.flattenState initial).mem alloc.flat (alloc.addr "o" 16) = _
    rw [alloc.flattenState_mem_addr disjoint initial (by simp [alloc]) (by decide)]
    rfl
  · rintro ⟨_,idx,ha,heq⟩
    dsimp [alloc,regions,layout,outOffset,tIndex,vIndex,initial,FlatAlloc.addr] at heq ha
    norm_num at heq
    omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ChunkGLAOutputReal
