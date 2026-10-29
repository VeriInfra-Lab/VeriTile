import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.FPValues

namespace VeriTile.Bench.Repairs.RmsnormTriton.Symbolic
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Region addresses begin at the actual tensor data pointer. The shared
allocation memory preserves input/weight aliases and nonzero storage offsets. -/
def state (p : Parameters) (before : Memory Term) (u : Nat → Term) (b m : Nat) : State Term where
  mem := fun r o=>.mk .real (
    if r="x" then before p.input.allocation (p.input.base+o)
    else if r="weights" then before p.weights.allocation (p.weights.base+o)
    else if r="out" then before p.output.allocation (p.output.base+o)
    else .unused)
  regs := fun _ _ _=>none
  pids := fun ax=>if ax=0 then b else if ax=1 then m else 0
  numPids := fun ax=>if ax=0 then p.batches else if ax=1 then p.rows else 1
  undef := fun d _ o=>match d with | .real=>u o | d=>defaultValue d

def result (p : Parameters) (before : Memory Term) (final : State Term) : Memory Term :=
  fun a o=>if a=p.output.allocation ∧ p.output.base≤o then read final "out" (o-p.output.base)
    else before a o

def rowModel (p : Parameters) (b m : Nat) (u : Nat → Term) (before : Memory Term) : Memory Term :=
  result p before (graph (state p before u b m) p blockSize)

/-- This executes the actual typed source. Failure is preserved by `Option`. -/
def rowRun (p : Parameters) (b m : Nat) (u : Nat → Term) (before : Memory Term) : Option (Memory Term) :=
  (FP.Structural.exec algebra (TensorSource.source p) (state p before u b m)).map (result p before)

theorem row_run (p : Parameters) (b m : Nat) (u : Nat → Term) (before : Memory Term) :
    rowRun p b m u before=some (rowModel p b m u before) := by
  obtain ⟨final,ran,mem⟩ := complete (state p before u b m) p blockSize
  change (run algebra (code p blockSize) (state p before u b m)).map _=some _
  rw [ran,Option.map_some]
  congr 1
  funext a o
  simp only [result,rowModel,read,mem]

theorem row_unique (p : Parameters) (b m : Nat) (hb : b<p.batches) (hm : m<p.rows)
    (unique : OutputUnique p) : RowUnique p b m := by
  intro k j hk hj same
  apply (unique b m k b m j hb hm hk hb hm hj ?_).2.2
  simpa [outputOffset,TensorSource.rowAddress,Nat.add_assoc] using congrArg (p.output.base+·) same

theorem result_read (p : Parameters) (before : Memory Term) (final : State Term) (b m k : Nat) :
    result p before final p.output.allocation (outputOffset p b m k)=
      read final "out" (TensorSource.rowAddress p b m k) := by
  have eq : outputOffset p b m k=p.output.base+TensorSource.rowAddress p b m k := by
    simp [outputOffset,TensorSource.rowAddress,Nat.add_assoc]
  simp [result,eq]

theorem result_frame (p : Parameters) (before : Memory Term) (final : State Term) (b m : Nat)
    (u : Nat → Term)
    (frame : ∀ o,(∀ k,k<p.width → o≠TensorSource.rowAddress p b m k) →
      final.mem "out" o=(state p before u b m).mem "out" o)
    (a o : Nat) (outside : ∀ k,k<p.width → ¬(a=p.output.allocation ∧ o=outputOffset p b m k)) :
    result p before final a o=before a o := by
  unfold result
  split_ifs with active
  · rcases active with ⟨rfl,bound⟩
    have fr := frame (o-p.output.base) (by
      intro k hk same
      apply outside k hk
      refine ⟨rfl,?_⟩
      simp only [outputOffset,TensorSource.rowAddress] at same ⊢
      omega)
    unfold read
    rw [fr]
    simp [state,Nat.add_sub_of_le bound]
  · rfl

def rowLogical (p : Parameters) (before : Memory Term) (b m k : Nat) : Term :=
  logical p.width blockSize p.eps
    (fun j=>before p.input.allocation (inputOffset p b m j))
    (fun j=>before p.weights.allocation (weightOffset p j)) k

theorem row_spec (p : Parameters) (b m : Nat) (hb : b<p.batches) (hm : m<p.rows)
    (unique : OutputUnique p) (u : Nat → Term) (before : Memory Term) :
    (∀ k,k<p.width → rowModel p b m u before p.output.allocation (outputOffset p b m k)=
      rowLogical p before b m k) ∧
    (∀ a o,(∀ k,k<p.width → ¬(a=p.output.allocation ∧ o=outputOffset p b m k)) →
      rowModel p b m u before a o=before a o) := by
  obtain ⟨values,frame⟩ := graph_spec (state p before u b m) p b m blockSize
    (by simp [state]) (by simp [state]) (by decide) (row_unique p b m hb hm unique)
  constructor
  · intro k hk
    rw [rowModel,result_read]
    unfold read
    rw [values k hk,Cell.read_same]
    simp [rowLogical,read,state,inputAddress,inputOffset,weightOffset,Nat.add_assoc]
  · intro a o outside
    exact result_frame p before _ b m u (fun o ho=>frame "out" o (Or.inr ho)) a o outside

theorem row_readonly (p : Parameters) (b m : Nat) (hb : b<p.batches) (hm : m<p.rows)
    (unique : OutputUnique p) (u : Nat → Term) (before : Memory Term) (a : Nat)
    (different : a≠p.output.allocation) : rowModel p b m u before a=before a := by
  funext o
  exact (row_spec p b m hb hm unique u before).2 a o (fun _ _ hit=>different hit.1)

def gridModel (p : Parameters) (u : Nat → Term) : List (Nat × Nat) → Memory Term → Memory Term
  | [],before=>before
  | (b,m)::rest,before=>gridModel p u rest (rowModel p b m u before)

def grid (p : Parameters) (u : Nat → Term) : List (Nat × Nat) → Memory Term → Option (Memory Term)
  | [],before=>some before
  | (b,m)::rest,before=>(rowRun p b m u before).bind (grid p u rest)

theorem grid_run (p : Parameters) (u : Nat → Term) (order : List (Nat × Nat)) (before : Memory Term) :
    grid p u order before=some (gridModel p u order before) := by
  induction order generalizing before with
  | nil=>rfl
  | cons bm rest ih=>
    rcases bm with ⟨b,m⟩
    simp only [grid,gridModel,row_run,Option.bind_some,ih]

theorem grid_frame (p : Parameters) (unique : OutputUnique p) (u : Nat → Term)
    (order : List (Nat × Nat)) (valid : ∀ bm∈order,bm.1<p.batches ∧ bm.2<p.rows)
    (before : Memory Term) (a o : Nat)
    (outside : ∀ bm∈order,∀ k,k<p.width → ¬(a=p.output.allocation ∧ o=outputOffset p bm.1 bm.2 k)) :
    gridModel p u order before a o=before a o := by
  induction order generalizing before with
  | nil=>rfl
  | cons bm rest ih=>
    rcases bm with ⟨b,m⟩
    have hv := valid (b,m) List.mem_cons_self
    rw [gridModel,ih (fun bm h=>valid bm (List.mem_cons_of_mem _ h)) _
      (fun bm h=>outside bm (List.mem_cons_of_mem _ h))]
    exact (row_spec p b m hv.1 hv.2 unique u before).2 a o (outside (b,m) List.mem_cons_self)

theorem grid_value (p : Parameters) (sep : Separation p) (unique : OutputUnique p) (u : Nat → Term)
    (order : List (Nat × Nat)) (valid : ∀ bm∈order,bm.1<p.batches ∧ bm.2<p.rows)
    (before : Memory Term) (b m k : Nat) (member : (b,m)∈order) (hk : k<p.width) :
    gridModel p u order before p.output.allocation (outputOffset p b m k)=rowLogical p before b m k := by
  induction order generalizing before with
  | nil=>simp at member
  | cons bm rest ih=>
    rcases bm with ⟨b',m'⟩
    have active := valid (b',m') List.mem_cons_self
    have target := valid (b,m) member
    rw [gridModel]
    by_cases tail : (b,m)∈rest
    · rw [ih (fun bm h=>valid bm (List.mem_cons_of_mem _ h)) _ tail]
      simp only [rowLogical,row_readonly p b' m' active.1 active.2 unique u before _ sep.input,
        row_readonly p b' m' active.1 active.2 unique u before _ sep.weights]
    · have eq : (b,m)=(b',m') := (List.mem_cons.mp member).resolve_right tail
      cases eq
      rw [grid_frame p unique u rest (fun bm h=>valid bm (List.mem_cons_of_mem _ h))]
      · exact (row_spec p b m target.1 target.2 unique u before).1 k hk
      · intro bm hbm j hj same
        have h := valid bm (List.mem_cons_of_mem _ hbm)
        have eq := unique b m k bm.1 bm.2 j target.1 target.2 hk h.1 h.2 hj same.2
        have pair : bm=(b,m) := Prod.ext eq.1.symm eq.2.1.symm
        exact tail (pair ▸ hbm)

theorem grid_result (p : Parameters) (sep : Separation p) (unique : OutputUnique p)
    (u : Nat → Term) (schedule : Schedule p) (before : Memory Term) :
    (∀ b m k,b<p.batches → m<p.rows → k<p.width →
      gridModel p u schedule.order before p.output.allocation (outputOffset p b m k)=rowLogical p before b m k) ∧
    (∀ a o,Outside p a o → gridModel p u schedule.order before a o=before a o) := by
  have valid : ∀ bm∈schedule.order,bm.1<p.batches ∧ bm.2<p.rows := by
    intro bm member
    exact (programGrid_mem p bm.1 bm.2).mp (schedule.complete.mem_iff.mp member)
  constructor
  · intro b m k hb hm hk
    exact grid_value p sep unique u schedule.order valid before b m k
      (schedule.complete.mem_iff.mpr ((programGrid_mem p b m).mpr ⟨hb,hm⟩)) hk
  · intro a o outside
    exact grid_frame p unique u schedule.order valid before a o
      (fun bm h k hk=>outside bm.1 bm.2 k (valid bm h).1 (valid bm h).2 hk)

#axiomsClean row_run
#axiomsClean grid_run
#axiomsClean grid_result
end
end VeriTile.Bench.Repairs.RmsnormTriton.Symbolic
