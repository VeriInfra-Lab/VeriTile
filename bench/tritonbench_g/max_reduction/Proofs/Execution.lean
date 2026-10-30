import bench.tritonbench_g.max_reduction.Proofs.Compute
import bench.tritonbench_g.max_reduction.Proofs.Grid
import bench.tritonbench_g.apply_penalty.Proofs.Scatter

namespace VeriTile.Bench.Repairs.MaxReduction.Execution
open Compute
open VeriTile.Bench.Repairs.ApplyPenalty
noncomputable section
open Classical
set_option maxHeartbeats 2000000

/-- Fresh Torch allocations have distinct region identities. A cell retains
its value or signed-int64 payload; untouched cells retain their original tag. -/
inductive Cell (A : Type) where
  | value (x : A)
  | index (x : IndexWord)
  | raw (x : Nat)
  deriving DecidableEq
abbrev Heap (A : Type) := String × Nat → Cell A

def read (padding : A) (s : Heap A) (region : String) (i : Nat) : A :=
  match s (region,i) with | .value x => x | _ => padding

def writeValue (s : Heap A) (region : String) (i : Nat) (x : A) : Heap A :=
  Scatter.write s (region,i) (.value x)

def writeIndex (s : Heap A) (region : String) (i : Nat) (x : IndexWord) : Heap A :=
  Scatter.write s (region,i) (.index x)

theorem read_write_other (padding : A) (s : Heap A) (r out : String) (i : Nat) (x : A)
    (h : r≠out) : read padding (writeValue s out i x) r=read padding s r := by
  funext j
  simp [read,writeValue,Scatter.write,h]

def scalarStep (algebra : Algebra A) (tree : Tree) (inp out : String)
    (count start output : Nat) (s : Heap A) : Heap A :=
  writeValue s out output (scalar algebra tree (read algebra.padding s inp) count start)

def scalarSteps (algebra : Algebra A) (trees : Nat → Tree) (inp out : String)
    (count block : Nat) : List Nat → Heap A → Heap A
  | [],s => s
  | p::rest,s => scalarSteps algebra trees inp out count block rest
      (scalarStep algebra (trees p) inp out count (p*block) p s)

/-- Current-heap reads reduce to entry reads because the only writes target a
fresh, distinct output region. No input preservation or reduction result is a premise. -/
theorem scalar_scatter (algebra : Algebra A) (trees : Nat → Tree) (inp out : String)
    (count block : Nat) (jobs : List Nat) (s : Heap A) (apart : inp≠out) :
    scalarSteps algebra trees inp out count block jobs s=
      Scatter.run jobs (fun p => (out,p))
        (fun p => Cell.value (scalar algebra (trees p) (read algebra.padding s inp) count (p*block))) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons p rest ih =>
    rw [scalarSteps,ih]
    simp only [scalarStep]
    rw [read_write_other _ _ _ _ _ _ apart]
    rfl

def globalRun (algebra : Algebra A) (count block width : Nat)
    (first : Nat → Tree) (second : Tree) (s : Heap A) : Heap A :=
  let reached := scalarSteps algebra first "input" "middle" count block
    (List.range (Address.count count block)) s
  scalarStep algebra second "middle" "output" (Address.count count block) 0 0 reached

def globalReference (algebra : Algebra A) (input : Nat → A) (count block : Nat)
    (first : Nat → Tree) (second : Tree) : A :=
  scalar algebra second (fun p => scalar algebra (first p) input count (p*block))
    (Address.count count block) 0

theorem global_result (algebra : Algebra A) (count block width : Nat)
    (first : Nat → Tree) (second : Tree) (s : Heap A) :
    globalRun algebra count block width first second s ("output",0)=
      .value (globalReference algebra (read algebra.padding s "input") count block first second) := by
  let reached := scalarSteps algebra first "input" "middle" count block
    (List.range (Address.count count block)) s
  have hm : ∀ p,p<Address.count count block → read algebra.padding reached "middle" p=
      scalar algebra (first p) (read algebra.padding s "input") count (p*block) := by
    intro p hp
    have hread := Scatter.readback_injective (List.range (Address.count count block))
      (fun p => ("middle",p))
      (fun p => Cell.value (scalar algebra (first p) (read algebra.padding s "input") count (p*block))) s
      p (List.mem_range.mpr hp) (by intro i j h; exact (Prod.mk.inj h).2)
    rw [← scalar_scatter algebra first "input" "middle" count block _ s (by decide)] at hread
    change reached ("middle",p)=Cell.value _ at hread
    simp only [read,hread]
  change writeValue reached "output" 0 _ ("output",0)=_
  simp only [writeValue,Scatter.write_same]
  congr 1
  exact scalar_local algebra second _ _ _ 0 hm

theorem global_frame (algebra : Algebra A) (count block width : Nat)
    (first : Nat → Tree) (second : Tree) (s : Heap A) (q : String × Nat)
    (middle : q.1≠"middle" ∨ Address.count count block≤q.2)
    (output : q≠("output",0)) : globalRun algebra count block width first second s q=s q := by
  simp only [globalRun,scalarStep,writeValue,Scatter.write,if_neg output]
  rw [scalar_scatter algebra first "input" "middle" count block _ s (by decide)]
  apply Scatter.frame
  intro p hp same
  have hs := congrArg Prod.fst same
  have hi := congrArg Prod.snd same
  have hp := List.mem_range.mp hp
  rcases middle with h|h
  · exact h hs
  · omega

def rowStep (algebra : Algebra A) (tree : Tree) (N K j : Nat) (s : Heap A) : Heap A :=
  let result := Compute.dimension algebra tree (read algebra.padding s "input") N K j
  writeIndex (writeValue s "values" j result.1) "indices" j result.2

def rowSteps (algebra : Algebra A) (tree : Tree) (N K : Nat) : List Nat → Heap A → Heap A
  | [],s => s
  | j::rest,s => rowSteps algebra tree N K rest (rowStep algebra tree N K j s)

def pairScatter (values : Nat → A × IndexWord) (jobs : List Nat) (s : Heap A) : Heap A :=
  jobs.foldl (fun s j => writeIndex (writeValue s "values" j (values j).1) "indices" j (values j).2) s

theorem pair_input (values : Nat → A × IndexWord) (jobs : List Nat) (s : Heap A) (padding : A) :
    read padding (pairScatter values jobs s) "input"=read padding s "input" := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j rest ih =>
    rw [pairScatter,List.foldl_cons,← pairScatter,ih]
    funext i
    simp [read,writeIndex,writeValue,Scatter.write]

theorem rows_scatter (algebra : Algebra A) (tree : Tree) (N K : Nat) (jobs : List Nat) (s : Heap A) :
    rowSteps algebra tree N K jobs s=
      pairScatter (fun j => Compute.dimension algebra tree (read algebra.padding s "input") N K j) jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j rest ih =>
    rw [rowSteps,ih]
    have input : read algebra.padding (rowStep algebra tree N K j s) "input"=read algebra.padding s "input" := by
      funext i
      simp [rowStep,read,writeIndex,writeValue,Scatter.write]
    rw [input]
    rfl

theorem pair_result (values : Nat → A × IndexWord) (jobs : List Nat) (s : Heap A)
    (j : Nat) (hj : j∈jobs) :
    pairScatter values jobs s ("values",j)=.value (values j).1 ∧
    pairScatter values jobs s ("indices",j)=.index (values j).2 := by
  have stable : ∀ (tail : List Nat) (state : Heap A),
      state ("values",j)=.value (values j).1 → state ("indices",j)=.index (values j).2 →
      pairScatter values tail state ("values",j)=.value (values j).1 ∧
      pairScatter values tail state ("indices",j)=.index (values j).2 := by
    intro tail
    induction tail with
    | nil => intro s hv hi; exact ⟨hv,hi⟩
    | cons k rest ih =>
      intro state hv hi
      apply ih
      · by_cases h : j=k
        · subst k; simp [writeIndex,writeValue,Scatter.write]
        · simp [writeIndex,writeValue,Scatter.write,h,hv]
      · by_cases h : j=k
        · subst k; simp [writeIndex,writeValue,Scatter.write]
        · simp [writeIndex,writeValue,Scatter.write,h,hi]
  obtain ⟨before,after,eq⟩ := List.append_of_mem hj
  rw [eq,pairScatter,List.foldl_append,List.foldl_cons,←pairScatter]
  apply stable <;> simp [writeIndex,writeValue,Scatter.write]

theorem pair_frame (values : Nat → A × IndexWord) (jobs : List Nat) (s : Heap A) (q : String × Nat)
    (miss : ∀ j∈jobs,q≠("values",j) ∧ q≠("indices",j)) : pairScatter values jobs s q=s q := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j rest ih =>
    change pairScatter values rest _ q=s q
    rw [ih _ (fun k hk => miss k (by simp [hk]))]
    simp [writeIndex,writeValue,Scatter.write,(miss j (by simp)).1,(miss j (by simp)).2]

def dimensionRun (algebra : Algebra A) (tree : Tree) (M N K BM : Nat) (s : Heap A) : Heap A :=
  rowSteps algebra tree N K (Grid.jobs M K BM) s

theorem dimension_result (algebra : Algebra A) (tree : Tree) (M N K BM : Nat)
    (hb : 0<BM) (s : Heap A) (j : Nat) (hj : j<M*K) :
    dimensionRun algebra tree M N K BM s ("values",j)=
      .value (Compute.dimension algebra tree (read algebra.padding s "input") N K j).1 ∧
    dimensionRun algebra tree M N K BM s ("indices",j)=
      .index (Compute.dimension algebra tree (read algebra.padding s "input") N K j).2 := by
  rw [dimensionRun,rows_scatter]
  exact pair_result _ _ _ j ((Grid.mem_jobs _ _ _ _ hb).mpr hj)

theorem dimension_frame (algebra : Algebra A) (tree : Tree) (M N K BM : Nat)
    (hb : 0<BM) (s : Heap A) (q : String × Nat)
    (miss : q.1≠"values" ∧ q.1≠"indices" ∨ M*K≤q.2) :
    dimensionRun algebra tree M N K BM s q=s q := by
  rw [dimensionRun,rows_scatter]
  apply pair_frame
  intro j hj
  have bound := (Grid.mem_jobs _ _ _ _ hb).mp hj
  rcases miss with h|h
  · exact ⟨fun eq => h.1 (congrArg Prod.fst eq),fun eq => h.2 (congrArg Prod.fst eq)⟩
  · constructor <;> intro eq <;> have hs := congrArg Prod.snd eq <;> omega

#axiomsClean scalar_scatter
#axiomsClean global_result
#axiomsClean global_frame
#axiomsClean dimension_result
#axiomsClean dimension_frame
end
end VeriTile.Bench.Repairs.MaxReduction.Execution
