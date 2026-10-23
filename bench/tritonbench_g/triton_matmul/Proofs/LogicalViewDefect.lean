import bench.tritonbench_g.triton_matmul.Proofs.SourceScope
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import Mathlib.Data.Fintype.Card

/-! Aligned original-source logical-view defect. The observed Torch wrapper
forwards an unresolved negative view's unchanged raw pointer. Both matrices
have complete M/N/K tiles here; this counterexample does not use, validate, or
refute the separate TM-TAIL-CONTIGUITY candidate. Native GPU execution is not
claimed. The full typed source below runs in an explicit ideal dot model. -/
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.LogicalViewDefect
open VeriTile.Triton FP.Structural FPValue SourceScope
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 4000000

/-- Actual explicit FP16 host branch, exactly one full tile and one K block. -/
def p : Parameters := parameters "A" "B" "C" .half 128 256 64 64 1 256 1

def initial : State ℝ where
  mem := fun _ _ => Cell.mk .fp16 1
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def idealDotValue {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → ℝ) (b : TileIndex (batch++[k,n]) → ℝ) :
    TileIndex (batch++[m,n]) → ℝ :=
  match batch with
  | [] => fun i => ∑ j : Fin k, a (i.1,j,PUnit.unit)*b (j,i.2.1,PUnit.unit)
  | _::tail => fun i => idealDotValue (batch:=tail) (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2
termination_by batch.length

def idealDot : DotAcc ℝ where
  apply := fun _ _ _ _ _ {batch} {m} {k} {n} a b acc i => acc i+idealDotValue (batch:=batch) a b i

abbrev ideal := ContextAttnFwd.FPModel.model

theorem physical_value (i : CIndex p) : outputValue ideal idealDot p .half initial i=64 := by
  have ha (j : AIndex p) : aValue ideal p .half initial 0 j=1 := by
    have hj : j.2.1.val<64 := j.2.1.isLt
    simp [aValue,p,parameters,blockK,BK,hj,FPValue.read,initial,Cell.read]
  have hb (j : BIndex p) : bValue ideal p .half initial 0 j=1 := by
    have hj : j.1.val<64 := j.1.isLt
    simp [bValue,p,parameters,blockK,BK,hj,FPValue.read,initial,Cell.read]
  have hu (acc : CIndex p → ℝ) : update ideal idealDot p .half initial 0 acc i=acc i+64 := by
    unfold update
    simp only [idealDot,idealDotValue]
    simp_rw [ha,hb]
    norm_num [p,parameters,blockK,BK]
  have hc : p.numKBlocks=1 := rfl
  unfold outputValue accumulation
  rw [hc]
  simp only [foldRange,show 1≠(0 : Nat) by decide,show 0<(1 : Nat) by decide,↓reduceDIte,
    show ¬1<(1 : Nat) by decide]
  change update ideal idealDot p .half initial 0 (fun _ => 0) i=64
  rw [hu]
  simp

def lane : CIndex p := (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)

/-- Exact source hint operands in the witness: complete contiguous coordinate
ranges, with no row/column clamp or partial tile. This is not a general proof
of native hint consumption, and says nothing about the pending tail candidate. -/
theorem complete_tile_coordinates (i : Fin 128) (j : Fin 256) :
    aRow p initial i=i.val ∧ bCol p initial j=j.val := by
  simp [aRow,bCol,row,col,pidM,pidN,firstM,perGroup,blocksM,blocksN,groupSize,
    p,parameters,blockK,BM,BN,initial,i.isLt,j.isLt]

/-- Every witness access lies within the ordinary finite tensor allocation. -/
theorem aligned_access_bounds (a : AIndex p) (b : BIndex p) (c : CIndex p) :
    aOffset p initial 0 a<128*64 ∧ bOffset p initial 0 b<64*256 ∧
    cOffset p initial c<128*256 := by
  have ha : a.1.val<128 := a.1.isLt
  have hak : a.2.1.val<64 := a.2.1.isLt
  have hb : b.1.val<64 := b.1.isLt
  have hbn : b.2.1.val<256 := b.2.1.isLt
  have hc : c.1.val<128 := c.1.isLt
  have hcn : c.2.1.val<256 := c.2.1.isLt
  simp [aOffset,bOffset,cOffset,aRow,bCol,row,col,pidM,pidN,firstM,perGroup,blocksM,blocksN,groupSize,
    p,parameters,blockK,BM,BN,BK,initial,ha,hbn]
  omega

/-- Independent logical tensor product of lazy-negative A and ordinary B. -/
def logical : ℝ := mathematical 128 256 64 (fun _ _ => -1) (fun _ _ => 1)
  ⟨0,by decide⟩ ⟨0,by decide⟩

/-- Full actual one-program launch, not just a scalar arithmetic observation.
Raw A/B are +1, so output is +64; logical lazy-negative A requests -64. -/
theorem aligned_source_disagrees :
    grid 128 256=1 ∧
    ∃ final, FP.Structural.exec (backend ideal idealDot) (source p .half) initial=some final ∧
      final.mem "C" 0=Cell.mk .fp16 (64 : ℝ) ∧ logical=(-64 : ℝ) ∧ (64 : ℝ)≠logical := by
  refine ⟨by decide,?_⟩
  obtain ⟨final,he,_,hv,hf⟩ := correctness ideal idealDot p .half initial
    (active_injective "A" "B" "C" .half 128 256 64 64 1 256 1 initial)
  have h := hv lane (by decide)
  rw [physical_value] at h
  have hn : logical=(-64 : ℝ) := by norm_num [logical,mathematical]
  refine ⟨final,he,?_,hn,?_⟩
  · simpa [stored,cOffset,row,col,pidM,pidN,firstM,perGroup,blocksM,blocksN,groupSize,
      p,parameters,blockK,BM,BN,initial,lane] using h
  · rw [hn]
    norm_num

#axiomsClean aligned_access_bounds
#axiomsClean complete_tile_coordinates
#axiomsClean physical_value
#axiomsClean aligned_source_disagrees
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.LogicalViewDefect
