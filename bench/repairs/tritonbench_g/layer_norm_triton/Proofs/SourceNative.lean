import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelNative
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceForwardResult
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedExecution
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceFinalGraph
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceStrideRefinement

/-! The actual typed source is interpreted into expression trees before any
native rounding. Only structural shape selects reductions, operand ordering,
constant specialization and adjacent FMA. Physical kernel boundaries are
materialized by SourceNativeStorage, so contractions cannot cross a store/load. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceNative
open VeriTile.Triton FP.Structural SourceModel
namespace Native
export VeriTile.Bench.Repairs.FastLayernorm.KernelNative
  (Word Shape Term Reductions Instructions Schedule OutputSchedule algebra reduction normalize eval observe)
end Native
open Native
noncomputable section

def model (R : Reductions) : Model Term where
  scalar := algebra R
  sumTyped := fun p d {_} ax keep xs out => ofFloat d
    (reduction R Bool.false p ax keep (fun i => toFloat d (xs i)) out)

/-- Every symbolic load starts from an actual word in the incoming memory.
Numerical payloads are excluded from the compiler's leaf shape. -/
def seed (s : State Word) : State Term where
  mem := fun region offset => .mk .real (.leaf region offset ((s.mem region offset).read .real))
  regs := fun _ _ _ => none
  pids := s.pids
  numPids := s.numPids
  undef := fun d _ _ => defaultValue d

def read (I : Instructions) (S : OutputSchedule) (s : State Term) (region : RegionName) (offset : Nat) : Word :=
  observe I (S region offset) (s.mem region offset)

theorem read_congr (I : Instructions) (S : OutputSchedule) (s t : State Term) (h : s.mem=t.mem)
    (region : RegionName) (offset : Nat) : read I S s region offset=read I S t region offset := by
  simp only [read,h]

theorem forward (R : Reductions) (stride N B : Nat) (eps : ℝ) (s : State Word)
    (hn : 0<N) (hb : N≤B) (hstride : s.pids 0*stride=s.pids 0*N) :
    ∃ t, FP.Structural.exec (engine (model R))
      (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) (seed s)=some t ∧
      t.mem=(SourceForwardGraph.final (model R) N N B eps (seed s)).mem := by
  obtain ⟨t,ht⟩ := SourceForward.source_executes (model R) stride N B eps hn hb (seed s)
  refine ⟨t,ht,(SourceForwardGraph.source_memory (model R) stride N B eps hn hb _ _ ht).trans ?_⟩
  exact SourceStrideRefinement.forward_memory (model R) (seed s) stride N B eps hstride

theorem grouped (R : Reductions) (stride M N G B : Nat) (s : State Word)
    (hstride : ∀ row,row<M → row*stride=row*N) :
    ∃ t, FP.Structural.exec (engine (model R))
      (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B)
      (seed s)=some t ∧ t.mem=(SourceGroupedExecution.final (model R) N M N G B (seed s)).mem := by
  obtain ⟨t,ht,hm⟩ := SourceGroupedExecution.source_refines (model R) stride M N G B (seed s)
  exact ⟨t,ht,hm.trans (SourceStrideRefinement.grouped_memory (model R) stride M N G B (seed s) hstride)⟩

theorem final (R : Reductions) (M N : Nat) (s : State Word) :
    ∃ t, FP.Structural.exec (engine (model R))
      (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N 32 128) (seed s)=some t ∧
      t.mem=(SourceFinalGraph.final (model R) (seed s) M N 32 128).mem :=
  SourceFinalGraph.source_refines (model R) (seed s) M N 32 128

/-- Equality follows from equality of the actual source expression trees,
for every native instruction semantics and every structural compiler policy. -/
theorem forward_observed (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (stride N B : Nat) (eps : ℝ) (s : State Word) (hn : 0<N) (hb : N≤B)
    (hstride : s.pids 0*stride=s.pids 0*N) :
    ∃ t, FP.Structural.exec (engine (model R))
      (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) (seed s)=some t ∧
      ∀ region offset,read I S t region offset=
        read I S (SourceForwardGraph.final (model R) N N B eps (seed s)) region offset := by
  obtain ⟨t,ht,hm⟩ := forward R stride N B eps s hn hb hstride
  exact ⟨t,ht,read_congr I S _ _ hm⟩

#axiomsClean forward
#axiomsClean grouped
#axiomsClean final
#axiomsClean forward_observed
#specNonCircular model avoiding [FP.Structural.exec]
#specNonCircular read avoiding [FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceNative
