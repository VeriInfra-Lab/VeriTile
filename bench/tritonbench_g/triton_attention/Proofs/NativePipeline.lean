import bench.tritonbench_g.triton_attention.Proofs.NativeForwardComplete
import bench.tritonbench_g.triton_attention.Proofs.NativePreprocessGrid
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Selected
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 10000000

def Values (one : Bool) (c : Shape) (scale : ℝ) (owners : TritonAttention.Whole001.Native.Owners)
    (x : Target.Inputs) (mem : Grid.Memory Graph) : Prop :=
  TensorData c mem "Out" (Target.forward c one scale x owners).output ∧
  TensorData c mem "DQ" (fun h i d=>Target.query c one scale (Target.prepared c one scale x owners) h i d c.blocks) ∧
  TensorData c mem "DK" (Target.key c one scale (Target.prepared c one scale x owners)) ∧
  TensorData c mem "DV" (Target.value c one scale (Target.prepared c one scale x owners))

/-- Complete native interpretation of the same three DSL kernels. Saved
forward and preprocess values are derived; the only initialized scratch
premise is DQ zero. All schedules are complete legal CTA permutations. -/
theorem whole_execution_graph (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw)
    (mem : Grid.Memory Graph) (μ : RegionName → Nat → Graph) (hi : Inputs mem μ)
    (hz : ∀ i : Coord c,mem "DQ" (offset c i)=.mk .fp32 (.word 0)) :
    ∃ final,run one c scale owners fw pw bw mem=some final ∧ Values one c scale owners (input c μ) final ∧
      (∀ r o,¬Grid.Written c r o → final r o=mem r o) := by
  obtain ⟨f,fe,fi,fo⟩:=forward_grid one c scale owners fw hf mem μ hi
  have fEq : f=fw.foldl (fun m pid=>store one .forward c scale owners pid m) mem :=
    Option.some.inj (fe.symm.trans (Grid.run_eq_fold _ _ (execute_success one .forward c scale owners) fw mem))
  have fRead (r : RegionName) (o : Nat) (hr : r≠"Out" ∧ r≠"L" ∧ r≠"M") : f r o=mem r o := by
    rw [fEq]
    exact Grid.fold_frame _ fw mem r o (by intro pid hpid m;exact forwardStep_readonly one c scale owners pid m r o hr)
  have preData : PreData c f (input c μ) (Target.forward c one scale (input c μ) owners) := by
    constructor
    · intro i;exact (fo i).1
    · intro i;exact fi "DO" (by simp) (offset c i)
    · intro h i;exact (fo (h,i,⟨0,c.dim_pos⟩)).2.1
  obtain ⟨p,pe,pi,po⟩:=preprocess_grid one c scale owners pw hp f (input c μ) _ preData
  have pEq : p=pw.foldl (fun m pid=>store one .preprocess c scale owners pid m) f :=
    Option.some.inj (pe.symm.trans (Grid.run_eq_fold _ _ (execute_success one .preprocess c scale owners) pw f))
  have pRead (r : RegionName) (o : Nat) (hr : r≠"NewDO" ∧ r≠"Delta") : p r o=f r o := by
    rw [pEq]
    exact Grid.fold_frame _ pw f r o (by intro pid hpid m;exact preprocessStep_readonly one c scale owners pid m r o hr)
  have bData : BackwardData c p (Target.prepared c one scale (input c μ) owners) := by
    constructor
    · intro i;rw [pRead "Q" _ (by decide)];exact fi "Q" (by simp) (offset c i)
    · intro i;rw [pRead "K" _ (by decide)];exact fi "K" (by simp) (offset c i)
    · intro i;rw [pRead "V" _ (by decide)];exact fi "V" (by simp) (offset c i)
    · intro i;exact (po i).1
    · intro h i;rw [pRead "M" _ (by decide)];exact (fo (h,i,⟨0,c.dim_pos⟩)).2.2
    · intro h i;exact (po (h,i,⟨0,c.dim_pos⟩)).2
  have pZero : ∀ i : Coord c,p "DQ" (offset c i)=.mk .fp32 (.word 0) := by
    intro i
    rw [pRead "DQ" _ (by decide),fRead "DQ" _ (by decide)]
    exact hz i
  obtain ⟨b,be,bi,bo⟩:=backward_grid one c scale owners bw hb p _ bData pZero
  have whole : run one c scale owners fw pw bw mem=some b := by
    unfold run
    rw [fe]
    change (Grid.run (execute one .preprocess c scale owners) pw f).bind
      (Grid.run (execute one .backward c scale owners) bw)=some b
    rw [pe]
    exact be
  refine ⟨b,whole,?_,?_⟩
  · refine ⟨?_,fun i=>(bo i).1,fun i=>(bo i).2.1,fun i=>(bo i).2.2⟩
    intro i
    have bEq : b=bw.foldl (fun m pid=>store one .backward c scale owners pid m) p :=
      Option.some.inj (be.symm.trans (Grid.run_eq_fold _ _ (execute_success one .backward c scale owners) bw p))
    rw [bEq,Grid.fold_frame _ bw p "Out" _ (by intro pid hpid m;exact backwardStep_readonly one c scale owners pid m "Out" _ (by decide)),
      pRead "Out" _ (by decide)]
    exact (fo i).1
  · have finalEq : b=stores one c scale owners fw pw bw mem:=Option.some.inj (whole.symm.trans (run_success one c scale owners fw pw bw mem))
    rw [finalEq]
    exact stores_frame one c scale owners fw pw bw mem
      (Grid.FullSchedule.legal _ _ hf) (Grid.FullSchedule.legal _ _ hp) (Grid.FullSchedule.legal _ _ hb)

def wordInput (c : Shape) (μ : RegionName → Nat → Word) : Target.Inputs:=input c (fun r o=>.word (μ r o))
def WordValues (one : Bool) (c : Shape) (scale : ℝ) (owners : TritonAttention.Whole001.Native.Owners)
    (I : Prior.Instructions) (literal : ℝ → Word) (μ : RegionName → Nat → Word) (mem : Grid.Memory Word) : Prop :=
  ∀ i : Coord c,
    mem "Out" (offset c i)=.mk .fp32 (((Target.forward c one scale (wordInput c μ) owners).output i.1.val i.2.1.val i.2.2.val).eval I literal) ∧
    mem "DQ" (offset c i)=.mk .fp32 ((Target.query c one scale (Target.prepared c one scale (wordInput c μ) owners) i.1.val i.2.1.val i.2.2.val c.blocks).eval I literal) ∧
    mem "DK" (offset c i)=.mk .fp32 ((Target.key c one scale (Target.prepared c one scale (wordInput c μ) owners) i.1.val i.2.1.val i.2.2.val).eval I literal) ∧
    mem "DV" (offset c i)=.mk .fp32 ((Target.value c one scale (Target.prepared c one scale (wordInput c μ) owners) i.1.val i.2.1.val i.2.2.val).eval I literal)

/-- Scalar and fixed-K8 native instruction semantics are supplied directly;
there is no whole-kernel equivalence or result equality premise. -/
theorem whole_execution_words (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Prior.Instructions) (literal : ℝ → Word) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw)
    (mem : Grid.Memory Word) (μ : RegionName → Nat → Word)
    (hi : ∀ r∈(["Q","K","V","DO"] : List RegionName),∀ o,mem r o=.mk .fp32 (μ r o))
    (hz : ∀ i : Coord c,mem "DQ" (offset c i)=.mk .fp32 0) :
    ∃ final,nativeRun one c scale owners I literal fw pw bw mem=some final ∧ WordValues one c scale owners I literal μ final ∧
      (∀ r o,¬Grid.Written c r o → final r o=mem r o) := by
  obtain ⟨f,he,hv,hframe⟩:=whole_execution_graph one c scale owners fw pw bw hf hp hb (words mem) (fun r o=>.word (μ r o))
    (by intro r hr o;simp only [words,hi r hr o];rfl)
    (by intro i;simp only [words,hz i];rfl)
  refine ⟨evaluate I literal f,by simp only [nativeRun,he,Option.map_some],?_,?_⟩
  · intro i
    have out:=hv.1 i
    have dq:=hv.2.1 i
    have dk:=hv.2.2.1 i
    have dv:=hv.2.2.2 i
    simp only [evaluate,out,dq,dk,dv]
    exact ⟨rfl,rfl,rfl,rfl⟩
  · intro r o ho
    change mapCell _ (f r o)=_
    rw [hframe r o ho]
    exact congrFun (congrFun (evaluate_words I literal mem) r) o

#axiomsClean whole_execution_graph
#axiomsClean whole_execution_words
#specNonCircular WordValues avoiding [run,nativeRun,stores,Grid.execute,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Selected
