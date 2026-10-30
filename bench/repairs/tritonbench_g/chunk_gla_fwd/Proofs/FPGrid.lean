import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPStorage
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPGrid
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open FPStorage (Memory state exportMemory)
noncomputable section
variable {α : Type} [Inhabited α]

/-- One source program, with a fresh register file and the FP32 storage view
required by its declared pointer types. Export removes view tags only. -/
def invoke (M : Algebra α) (kernel : ComputeKernel) (grid : Grid) (idx : GridIndex grid)
    (m : Memory α) : Option (Memory α) :=
  (FP.Structural.exec (engine M) kernel (state m idx.toPids grid.toNumPids)).map exportMemory

def jobs (grid : Grid) : List (GridIndex grid) := Finset.univ.toList
theorem jobs_complete (grid : Grid) : (jobs grid).Nodup ∧ ∀ idx : GridIndex grid, idx∈jobs grid :=
  ⟨Finset.nodup_toList _,by simp [jobs]⟩

def runJobs (M : Algebra α) (kernel : ComputeKernel) (grid : Grid) :
    List (GridIndex grid) → Memory α → Option (Memory α)
  | [], m => some m
  | idx::rest, m => (invoke M kernel grid idx m).bind (runJobs M kernel grid rest)

/-- Independent numerical/address recurrence, with no interpreter query. -/
def referenceJobs {ι : Type} (step : ι → Memory α → Memory α) (order : List ι) (m : Memory α) : Memory α :=
  order.foldl (fun a i => step i a) m

theorem runJobs_eq (M : Algebra α) (kernel : ComputeKernel) (grid : Grid)
    (reference : GridIndex grid → Memory α → Memory α)
    (hstep : ∀ idx m, invoke M kernel grid idx m=some (reference idx m))
    (order : List (GridIndex grid)) (m : Memory α) :
    runJobs M kernel grid order m=some (referenceJobs reference order m) := by
  induction order generalizing m with
  | nil => rfl
  | cons idx rest ih => simp only [runJobs,hstep,Option.bind_some,referenceJobs,List.foldl_cons]; exact ih _

theorem reference_frame {ι : Type} (step : ι → Memory α → Memory α)
    (order : List ι) (m : Memory α) (r : RegionName) (off : Nat)
    (hstep : ∀ idx∈order, ∀ m, step idx m r off=m r off) :
    referenceJobs step order m r off=m r off := by
  induction order generalizing m with
  | nil => rfl
  | cons idx rest ih =>
    simp only [referenceJobs,List.foldl_cons]
    rw [show rest.foldl (fun a i => step i a) (step idx m) r off=step idx m r off from
      ih _ (fun j hj => hstep j (List.mem_cons_of_mem _ hj))]
    exact hstep idx (by simp) m

theorem reference_invariant {ι : Type} (step : ι → Memory α → Memory α)
    (order : List ι) (m : Memory α) (I : Memory α → Prop) (hI : I m)
    (hstep : ∀ idx∈order, ∀ m, I m → I (step idx m)) :
    I (referenceJobs step order m) := by
  induction order generalizing m with
  | nil => exact hI
  | cons idx rest ih =>
    apply ih (step idx m) (hstep idx (by simp) m hI)
    exact fun j hj => hstep j (List.mem_cons_of_mem _ hj)

theorem reference_readback {ι : Type} [DecidableEq ι] (step : ι → Memory α → Memory α)
    (order : List ι) (owner : ι) (howner : owner∈order) (hnodup : order.Nodup)
    (m : Memory α) (I : Memory α → Prop) (hI : I m) (r : RegionName) (off : Nat) (v : α)
    (hstep : ∀ idx∈order, ∀ m, I m → I (step idx m))
    (hwrite : ∀ m, I m → step owner m r off=v)
    (hframe : ∀ idx∈order, idx≠owner → ∀ m, step idx m r off=m r off) :
    referenceJobs step order m r off=v := by
  obtain ⟨before,after,he⟩ := List.append_of_mem howner
  subst order
  have hnot : owner∉after := (List.nodup_cons.mp (List.nodup_append.mp hnodup).2.1).1
  have hb : I (referenceJobs step before m) := reference_invariant step before m I hI
    (fun j hj => hstep j (List.mem_append_left _ hj))
  simp only [referenceJobs,List.foldl_append,List.foldl_cons]
  change referenceJobs step after (step owner (referenceJobs step before m)) r off=v
  exact (reference_frame step after _ r off (by
      intro j hj
      apply hframe j (by simp [hj])
      intro heq
      exact hnot (heq ▸ hj))).trans (hwrite _ hb)

#specNonCircular referenceJobs avoiding [runJobs,invoke,FP.Structural.exec]
#axiomsClean runJobs_eq
#axiomsClean reference_frame
#axiomsClean reference_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPGrid
