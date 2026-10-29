import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Proofs.IOGrid
import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Formulas

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section
set_option maxHeartbeats 1000000

theorem fold_select_last {ι C : Type} (items : List ι) (test : ι → Bool)
    (value : ι → C) (initial : C) :
    items.foldl (fun v i => if test i then value i else v) initial =
      ((items.filter test).getLast?.map value).getD initial := by
  induction items generalizing initial with
  | nil => rfl
  | cons i rest ih =>
    by_cases hi : test i=Bool.true
    · simp only [List.foldl_cons,hi,↓reduceIte,ih,List.filter_cons_of_pos hi]
      cases hr : rest.filter test with
      | nil => simp
      | cons j tail =>
        cases hl : (j::tail).getLast? <;> simp_all [List.getLast?_cons_cons]
    · have hf : test i=Bool.false := by cases h : test i <;> simp_all
      simp [List.foldl_cons,hf,ih]

theorem scatter_at {ι C : Type} (items : List ι) (region : RegionName)
    (off : ι → Nat) (value : ι → Cell C) (s : State C) (r : RegionName) (o : Nat) :
    (items.foldl (fun t i => t.write region (off i) (value i)) s).mem r o =
      items.foldl (fun v i => if r=region ∧ o=off i then value i else v) (s.mem r o) := by
  induction items generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons,ih,List.foldl_cons]
    rfl

variable {A B : Type} [Inhabited A] [Inhabited B]
variable (M : Algebra B) (dot : Dot B) (encode : A → B) (x : Layout) (scale : B)
  (observe : (port : Fin (outputs x).length) → TileIndex (outputs x)[port].shape → B → A)
  (xs : Inputs A x)

def jobWriters (x : Layout) (job : Program) : List (Logical.Writer x) :=
  ((TileShape.allIndices [128,x.shape.dim]).filter
    (fun i => decide (job.1*128+i.1.val<x.shape.length))).map (fun i => (job,i))

def writerValue (w : Logical.Writer x) : Cell B :=
  .mk .real (Logical.output M (parameters x) (Logical.rowAccumulator M dot encode x scale xs w.1) w.2)

theorem program_output_fold (job : Program) (valid : job.Valid x) (s : State B)
    (hin : (sourceIO M dot encode x scale observe).InputBinding xs s) (o : Nat) :
    referenceProgram M dot (parameters x) scale job s.mem "O" o =
      (jobWriters x job).foldl
        (fun v w => if o=Logical.writerAddress x w then writerValue M dot encode x scale xs w else v)
        (s.mem "O" o) := by
  have acc := accumulated_binding M dot encode x scale observe xs s hin job valid
  unfold referenceProgram written
  rw [acc,State.masked_scatter_eq_filter,scatter_at]
  rw [State.masked_scatter_frame]
  · simp only [jobWriters,List.foldl_map,true_and]
    rfl
  · exact Or.inl (by decide)

theorem grid_output_fold (jobs : List Program) (valid : ∀ job∈jobs,job.Valid x)
    (s : State B) (hin : (sourceIO M dot encode x scale observe).InputBinding xs s) (o : Nat) :
    referenceGrid M dot (parameters x) scale jobs s.mem "O" o =
      (jobs.flatMap (jobWriters x)).foldl
        (fun v w => if o=Logical.writerAddress x w then writerValue M dot encode x scale xs w else v)
        (s.mem "O" o) := by
  induction jobs generalizing s with
  | nil => rfl
  | cons job rest ih =>
    let next : State B := { s with mem := referenceProgram M dot (parameters x) scale job s.mem }
    have nextBinding : (sourceIO M dot encode x scale observe).InputBinding xs next :=
      binding_preserved M dot encode x scale observe xs s next hin [job] rfl
    change referenceGrid M dot (parameters x) scale rest next.mem "O" o=_
    rw [ih (fun j hj => valid j (by simp [hj])) next nextBinding]
    simp only [List.flatMap_cons,List.foldl_append]
    congr 1
    exact program_output_fold M dot encode x scale observe xs job (valid job (by simp)) s hin o

theorem grid_output_last (s : State B)
    (hin : (sourceIO M dot encode x scale observe).InputBinding xs s) (o : Nat) :
    referenceGrid M dot (parameters x) scale (programs (parameters x)) s.mem "O" o =
      ((Logical.lastWriter x o).map (writerValue M dot encode x scale xs)).getD (s.mem "O" o) := by
  rw [grid_output_fold M dot encode x scale observe xs _ (fun j hj => (program_valid x j).mp hj) s hin o]
  have selected := fold_select_last (Logical.writers x)
    (fun w => decide (o=Logical.writerAddress x w)) (writerValue M dot encode x scale xs) (s.mem "O" o)
  simpa [Logical.writers,Logical.lastWriter,jobWriters,eq_comm] using selected

#axiomsClean grid_output_last
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
