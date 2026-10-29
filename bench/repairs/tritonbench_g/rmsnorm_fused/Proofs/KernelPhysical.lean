import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.KernelBytes

namespace VeriTile.Bench.Repairs.RmsnormFused.Bytes
open VeriTile Triton FP.Structural Binding Ordered
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed (Term graphAlgebra)
noncomputable section
open Classical
set_option maxHeartbeats 6000000

def scatter {ι : Type} (xs : List ι) (offset : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v : ι → Term Word)
    (I : Instructions) (P : Policy) (R : Reductions) (N : Nat) (y : Tensor) (memory : Memory) : Memory :=
  xs.foldl (fun m i => if active i then
    store m (y.view.address (offset i)) (interpret I P R (offset i%N) (v i)) else m) memory

theorem scatter_commutes {ι : Type} (xs : List ι) (offset : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v : ι → Term Word)
    (I : Instructions) (P : Policy) (R : Reductions) (N : Nat) (y : Tensor) (memory : Memory)
    (s : State (Term Word)) (bounds : ∀ i,active i → offset i<y.size) :
    project I P R N y memory
      (xs.foldl (fun st i => if active i then st.write "Y" (offset i) (.mk .real (v i)) else st) s)=
      scatter xs offset active v I P R N y (project I P R N y memory s) := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons,scatter,List.foldl_cons]
    by_cases hi : active i
    · simp only [hi,if_true]
      rw [ih,write_commutes I P R N y memory s (offset i) (bounds i hi) (v i)]
      rfl
    · simp only [hi,if_false]
      exact ih s

def execute (I : Instructions) (P : Policy) (R : Reductions) (N B : Nat) (epsilon : Word)
    (y : Tensor) : List Nat → State (Term Word) → Memory → Memory
  | [],_,memory => memory
  | row::rest,s,memory => execute I P R N B epsilon y rest
      (Runtime.block graphAlgebra s "X" "Y" "W" row N B (.input epsilon))
      (scatter (TileShape.allIndices [B]) (fun i => row*N+i.1.val) (fun i => i.1.val<N)
        (fun i => Runtime.value graphAlgebra s "X" "W" row N B (.input epsilon) i.1.val)
        I P R N y memory)

theorem grid_commutes (I : Instructions) (P : Policy) (R : Reductions) (N B : Nat) (epsilon : Word)
    (y : Tensor) (schedule : List Nat) (s : State (Term Word)) (memory : Memory)
    (bounds : ∀ row∈schedule,∀ col,col<N → row*N+col<y.size) :
    project I P R N y memory (Runtime.grid graphAlgebra "X" "Y" "W" N B (.input epsilon) schedule s)=
      execute I P R N B epsilon y schedule s (project I P R N y memory s) := by
  induction schedule generalizing s with
  | nil => rfl
  | cons row rest ih =>
    rw [Runtime.grid,ih _ (fun r hr => bounds r (by simp [hr]))]
    change execute I P R N B epsilon y rest _ (project I P R N y memory
      ((TileShape.allIndices [B]).foldl _ s))=execute I P R N B epsilon y rest _ _
    rw [scatter_commutes _ _ _ _ I P R N y memory s (fun i hi => bounds row (by simp) i.1.val hi)]

/-- A successful actual source run is exactly a concrete sequence of physical
four-byte stores, starting at the caller's unchanged global byte memory. -/
theorem run_bytes (I : Instructions) (P : Policy) (R : Reductions)
    (x w y : Tensor) (eps : EpsilonInfo) (c : InputContract x.info w.info eps)
    (out : Output x w y) (epsilon : Word) (memory : Memory) (schedule : List Nat)
    (all : schedule.Perm (geometry x.info).grid) (final : State (Term Word))
    (ran : Runtime.run graphAlgebra "X" "Y" "W" (geometry x.info).rows (width x.info)
      (block (width x.info)) (.input epsilon) schedule (initial x w y memory)=some final) :
    project I P R (width x.info) y memory final=
      execute I P R (width x.info) (block (width x.info)) epsilon y schedule (initial x w y memory) memory := by
  obtain ⟨t,ht,hm⟩ := Runtime.source_complete graphAlgebra "X" "Y" "W" (geometry x.info).rows
    (width x.info) (block (width x.info)) (.input epsilon) c.widthPositive c.block schedule (initial x w y memory)
  have same : t=final := Option.some.inj (ht.symm.trans ran)
  subst t
  have projection : project I P R (width x.info) y memory final=
      project I P R (width x.info) y memory
        (Runtime.grid graphAlgebra "X" "Y" "W" (width x.info) (block (width x.info)) (.input epsilon)
          schedule (initial x w y memory)) := by
    unfold project reference Runtime.read
    rw [hm]
  rw [projection,grid_commutes,entry_project]
  intro row hr col hc
  rw [out.size,(input_sizes x w eps c).1]
  exact offset_bounded (geometry x.info) row col
    ((grid_complete _ _).mp (all.mem_iff.mp hr)) hc

theorem disjoint_read (y t : Tensor) (memory : Memory) (values : Nat → Word)
    (separated : y.view.base+4*y.size≤t.view.base ∨ t.view.base+4*t.size≤y.view.base)
    (k : Nat) (active : k<t.size) :
    t.read (load (reference y memory values)) k=t.read (load memory) k := by
  rw [t.read_view _ k active,t.read_view _ k active]
  unfold load
  congr 1
  funext byte
  apply reference_frame
  unfold footprint Torch.FP32.DataView.address
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.Bytes
