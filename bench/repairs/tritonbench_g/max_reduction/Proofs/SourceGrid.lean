import bench.repairs.tritonbench_g.max_reduction.Proofs.Source
import bench.repairs.tritonbench_g.max_reduction.Proofs.Machine

namespace VeriTile.Bench.Repairs.MaxReduction.SourceGrid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ApplyPenalty
noncomputable section
open Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

abbrev Flat (A : Type) := RegionName × Nat → FP.Structural.Cell A

def flatten (s : State A) : Flat A := fun q => s.mem q.1 q.2

def positioned (s : State A) (p k : Nat) : State A :=
  {s with pids := fun ax => if ax=0 then p else if ax=1 then k else 0}

@[simp] theorem positioned_flat (s : State A) (p k : Nat) : flatten (positioned s p k)=flatten s := rfl

theorem write_flat (s : State A) (r : RegionName) (i : Nat) (v : FP.Structural.Cell A) :
    flatten (s.write r i v)=Scatter.write (flatten s) (r,i) v := by
  funext q
  simp [flatten,State.write,Scatter.write,Prod.ext_iff]

theorem scatter_flat {ι : Type} (r : RegionName) (off : ι → Nat) (val : ι → FP.Structural.Cell A)
    (events : List ι) (s : State A) :
    flatten (events.foldl (fun t i => t.write r (off i) (val i)) s)=
      Scatter.run events (fun i => (r,off i)) val (flatten s) := by
  induction events generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons,ih,write_flat]
    rfl

def activeLanes (M BM : Nat) (p : Nat × Nat) : List (TileIndex [BM]) :=
  (TileShape.allIndices [BM]).filter fun i => p.1*BM+i.1.val<M

def rowJobs (M K BM : Nat) (p : Nat × Nat) : List Nat :=
  (activeLanes M BM p).map fun i => (p.1*BM+i.1.val)*K+p.2

def events (M K BM : Nat) (p : Nat × Nat) : List (Bool × Nat) :=
  (rowJobs M K BM p).map (fun j => (Bool.false,j)) ++ (rowJobs M K BM p).map (fun j => (Bool.true,j))

def key (event : Bool × Nat) : RegionName × Nat :=
  (if event.1 then "indices" else "values",event.2)

def target (e : Source.Engine A) (input : Nat → A) (N K BN j : Nat) : A × Nat :=
  e.indexed BN (fun n => if n.val<N then input (j/K*N*K+n.val*K+j%K) else e.padding)

def cell (values : Nat → A × Nat) (event : Bool × Nat) : FP.Structural.Cell A :=
  if event.1 then .mk .int ((values event.2).2 : Int) else .mk .real (values event.2).1

theorem key_injective : Function.Injective key := by
  rintro ⟨b,i⟩ ⟨c,j⟩ same
  cases b <;> cases c <;> simp_all [key,Prod.mk.injEq]

theorem row_target (e : Source.Engine A) (input : Nat → A) (M N K BM BN : Nat)
    (p : Nat × Nat) (kbound : p.2<K) (i : TileIndex [BM]) (active : p.1*BM+i.1.val<M) :
    Source.rowValue e input M N K BM BN p.1 p.2 i.1.val=
      target e input N K BN ((p.1*BM+i.1.val)*K+p.2) := by
  have hk : 0<K := by omega
  have hd : ((p.1*BM+i.1.val)*K+p.2)/K=p.1*BM+i.1.val := by
    rw [Nat.mul_comm _ K,Nat.mul_add_div hk,Nat.div_eq_of_lt kbound,Nat.add_zero]
  simp [Source.rowValue,target,hd,Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt kbound,active]

theorem fold_congr {ι β : Type} (events : List ι) (f g : β → ι → β) (s t : β)
    (initial : s=t) (same : ∀ i∈events,∀ state,f state i=g state i) :
    events.foldl f s=events.foldl g t := by
  subst t
  induction events generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [same i (by simp)]
    exact ih _ (fun j hj state => same j (by simp [hj]) state)

/-- Exact per-program AST execution is connected to its two masked store
streams. Cells and integer payloads retain the structural interpreter tags. -/
theorem dimension_program [Inhabited A] (e : Source.Engine A) (M N K BM BN : Nat)
    (p : Nat × Nat) (kbound : p.2<K) (hn : 0<BN) (s : State A) :
    ∃ t,FP.Structural.exec e.algebra (dimension "input" "values" "indices" M N K BM BN)
        (positioned s p.1 p.2)=some t ∧
      flatten t=Scatter.run (events M K BM p) key
        (cell (target e (fun i => (s.mem "input" i).read .real) N K BN)) (flatten s) := by
  obtain ⟨t,he,hm⟩ := Source.dimension_exec e "input" "values" "indices" M N K BM BN
    (positioned s p.1 p.2) hn
  refine ⟨t,he,?_⟩
  have hf : flatten t=flatten (Source.dimensionStores e "input" "values" "indices" M N K BM BN
      (positioned s p.1 p.2)) := congrArg (fun mem => fun q : RegionName × Nat => mem q.1 q.2) hm
  rw [hf]
  unfold Source.dimensionStores
  simp only [positioned,ite_true,show (1 : Nat)≠0 by decide,if_false]
  simp only [State.masked_scatter_eq_filter,scatter_flat]
  simp only [events,rowJobs,activeLanes,Scatter.run,List.foldl_append,List.foldl_map,key,cell,
    Bool.false_eq_true,if_false,if_true,positioned_flat,Function.comp_def]
  apply fold_congr
  · apply fold_congr
    · rfl
    · intro i hi mem
      rw [row_target e _ M N K BM BN p kbound i (of_decide_eq_true (List.mem_filter.mp hi).2)]
  · intro i hi mem
    rw [row_target e _ M N K BM BN p kbound i (of_decide_eq_true (List.mem_filter.mp hi).2)]

def runDimension [Inhabited A] (e : Source.Engine A) (M N K BM BN : Nat) :
    List (Nat × Nat) → State A → Option (State A)
  | [],s => some s
  | p::rest,s => do
      let t ← FP.Structural.exec e.algebra (dimension "input" "values" "indices" M N K BM BN)
        (positioned s p.1 p.2)
      runDimension e M N K BM BN rest t

theorem dimension_chain [Inhabited A] (e : Source.Engine A) (M N K BM BN : Nat) (hn : 0<BN)
    (order : List (Nat × Nat)) (bounds : ∀ p∈order,p.2<K) (s : State A) :
    ∃ t,runDimension e M N K BM BN order s=some t ∧
      flatten t=Scatter.run (order.flatMap (events M K BM)) key
        (cell (target e (fun i => (s.mem "input" i).read .real) N K BN)) (flatten s) := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons p rest ih =>
    obtain ⟨mid,hm,hmem⟩ := dimension_program e M N K BM BN p (bounds p (by simp)) hn s
    obtain ⟨t,ht,htmem⟩ := ih (fun q hq => bounds q (by simp [hq])) mid
    have input : (fun i => (mid.mem "input" i).read .real)=(fun i => (s.mem "input" i).read .real) := by
      funext i
      have frame : flatten mid ("input",i)=flatten s ("input",i) := by
        rw [hmem]
        apply Scatter.frame
        intro event _ eq
        have := congrArg (fun q => q.1.name) eq
        cases event with | mk flag j => cases flag <;> simp [key] at this
      exact congrArg (FP.Structural.Cell.read .real) frame
    refine ⟨t,by simp only [runDimension,hm]; exact ht,?_⟩
    rw [htmem,input,hmem]
    simp only [List.flatMap_cons,Scatter.run,List.foldl_append]

/-- Every output cell is written by the actual complete program grid. -/
theorem event_coverage (M K BM : Nat) (hb : 0<BM) (flag : Bool) (j : Nat) :
    (flag,j)∈(Grid.programs M K BM).flatMap (events M K BM) ↔ j<M*K := by
  simp only [List.mem_flatMap]
  constructor
  · rintro ⟨p,hp,he⟩
    have hj : j∈rowJobs M K BM p := by cases flag <;> simpa [events] using he
    obtain ⟨i,hi,rfl⟩ := List.mem_map.mp hj
    have hm : p.1*BM+i.1.val<M := of_decide_eq_true (List.mem_filter.mp hi).2
    exact Address.output_bound M K _ _ hm ((Grid.mem_programs _ _ _ _ _).mp hp).2
  · intro hj
    have hk : 0<K := by nlinarith
    have hm := (Nat.div_lt_iff_lt_mul hk).mpr hj
    obtain ⟨hp,hl,he⟩ := Address.segment_owner M BM (j/K) hb hm
    have hi : j/K%BM<BM := Nat.mod_lt _ hb
    let i : TileIndex [BM] := (⟨j/K%BM,hi⟩,PUnit.unit)
    let p := (j/K/BM,j%K)
    refine ⟨p,(Grid.mem_programs _ _ _ _ _).mpr ⟨hp,Nat.mod_lt _ hk⟩,?_⟩
    have lane : i∈activeLanes M BM p := by
      simp only [activeLanes,List.mem_filter,decide_eq_true_eq]
      refine ⟨TileShape.mem_allIndices _ i,?_⟩
      change j/K/BM*BM+j/K%BM<M
      have he' : j/K/BM*BM+j/K%BM=j/K := he
      rw [he']
      exact hm
    have job : j∈rowJobs M K BM p := by
      apply List.mem_map.mpr
      refine ⟨i,lane,?_⟩
      change (j/K/BM*BM+j/K%BM)*K+j%K=j
      have he' : j/K/BM*BM+j/K%BM=j/K := he
      rw [he']
      simpa [Nat.mul_comm] using Nat.div_add_mod j K
    cases flag <;> simp [events,job]

/-- Whole AST grid, successful execution, every value and signed integer
output, and full typed-cell frame. This theorem has no whole-output premise. -/
theorem dimension_whole [Inhabited A] (e : Source.Engine A) (M N K BM BN : Nat)
    (hb : 0<BM) (hn : 0<BN) (s : State A) :
    ∃ t,runDimension e M N K BM BN (Grid.programs M K BM) s=some t ∧
      (∀ j,j<M*K →
        t.mem "values" j=.mk .real (target e (fun i => (s.mem "input" i).read .real) N K BN j).1 ∧
        t.mem "indices" j=.mk .int ((target e (fun i => (s.mem "input" i).read .real) N K BN j).2 : Int)) ∧
      (∀ (r : RegionName) (i : Nat), (r ≠ "values" ∧ r ≠ "indices" ∨ M*K ≤ i) → t.mem r i=s.mem r i) := by
  obtain ⟨t,he,hm⟩ := dimension_chain e M N K BM BN hn (Grid.programs M K BM)
    (fun p hp => ((Grid.mem_programs _ _ _ _ _).mp hp).2) s
  refine ⟨t,he,?_,?_⟩
  · intro j hj
    have value := Scatter.readback_injective _ key
      (cell (target e (fun i => (s.mem "input" i).read .real) N K BN)) (flatten s)
      (Bool.false,j) ((event_coverage _ _ _ hb Bool.false j).mpr hj) key_injective
    have index := Scatter.readback_injective _ key
      (cell (target e (fun i => (s.mem "input" i).read .real) N K BN)) (flatten s)
      (Bool.true,j) ((event_coverage _ _ _ hb Bool.true j).mpr hj) key_injective
    rw [←hm] at value index
    exact ⟨value,index⟩
  · intro r i miss
    change flatten t (r,i)=flatten s (r,i)
    rw [hm]
    apply Scatter.frame
    rintro ⟨flag,j⟩ hj eq
    have active := (event_coverage M K BM hb flag j).mp hj
    have ri := congrArg Prod.snd eq
    have rr := congrArg Prod.fst eq
    rcases miss with h|h
    · cases flag <;> simp [key] at rr <;> tauto
    · simp [key] at ri
      omega

#axiomsClean dimension_program
#axiomsClean dimension_chain
#axiomsClean dimension_whole
end
end VeriTile.Bench.Repairs.MaxReduction.SourceGrid
