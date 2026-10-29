import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.Runtime
import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.KernelBytes

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Bytes
open VeriTile Triton FP.Structural Binding Ordered
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000

/-- The two source stores execute all real lanes before all imaginary lanes. -/
def scatter (xs : List Lane)
    (p : Primitives) (r : Request) (y : Tensor) (s : State Tagged) (im : Bool) (memory : Memory) : Memory :=
  xs.foldl (fun m i => if Source.active (parameters r "X" "Y") s im i then
    store m (y.view.address (Source.outputOffset (parameters r "X" "Y") s im i))
      (Source.value (operations p).backend (parameters r "X" "Y") s im i).word else m) memory

theorem scatter_commutes (p : Primitives) (r : Request) (y : Tensor) (memory : Memory)
    (s current : State Tagged) (im : Bool) (xs : List Lane)
    (bounds : ∀ i,Source.active (parameters r "X" "Y") s im i →
      Source.outputOffset (parameters r "X" "Y") s im i<y.size) :
    project y memory (xs.foldl (fun st i => if Source.active (parameters r "X" "Y") s im i then
      st.write "Y" (Source.outputOffset (parameters r "X" "Y") s im i)
        (.mk .fp32 (Source.value (operations p).backend (parameters r "X" "Y") s im i)) else st) current)=
      scatter xs p r y s im (project y memory current) := by
  induction xs generalizing current with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons,scatter,List.foldl_cons]
    by_cases hi : Source.active (parameters r "X" "Y") s im i
    · simp only [hi,if_true]
      rw [ih,write_commutes y memory current _ (bounds i hi)]
      rfl
    · simp only [hi,if_false]
      exact ih current

def execute (p : Primitives) (r : Request) (y : Tensor) :
    List Runtime.Program → State Tagged → Memory → Memory
  | [],_,memory => memory
  | q::rest,s,memory =>
    let entry := Runtime.setup s q
    let first := scatter (TileShape.allIndices [2,1024/2]) p r y entry Bool.false memory
    let second := scatter (TileShape.allIndices [2,1024/2]) p r y entry Bool.true first
    execute p r y rest (Runtime.block (operations p) r "X" "Y" q s) second

theorem block_bounds (r : Request) (y : Tensor) (size : y.size=r.size)
    (q : Runtime.Program) (hq : q.1<r.batches) (s : State Tagged)
    (im : Bool) (i : Source.Index (parameters r "X" "Y"))
    (hi : Source.active (parameters r "X" "Y") (Runtime.setup s q) im i) :
    Source.outputOffset (parameters r "X" "Y") (Runtime.setup s q) im i<y.size := by
  have active : active r (Runtime.location q i im) := by
    refine ⟨hq,hi.1,?_⟩
    cases im <;> simpa [Source.active,column,Source.column,Source.row,Runtime.setup,Runtime.location,
      row,evenColumn,program,lane,imaginary,parameters,Source.parts,parts,Nat.add_comm] using hi.2
  have off : Source.outputOffset (parameters r "X" "Y") (Runtime.setup s q) im i=
      r.offset (batch (Runtime.location q i im)) (row r (Runtime.location q i im))
        (column r (Runtime.location q i im)) :=
    (source_offset r "X" "Y" (Runtime.setup s q) (Runtime.location q i im)).2
  rw [off,size]
  exact offset_lt r _ _ _ active.1 active.2.1 active.2.2

theorem compose_commutes (p : Primitives) (r : Request) (y : Tensor) (size : y.size=r.size)
    (qs : List Runtime.Program) (valid : ∀ q∈qs,q.1<r.batches) (s : State Tagged) (memory : Memory) :
    project y memory (Runtime.compose (operations p) r "X" "Y" qs s)=
      execute p r y qs s (project y memory s) := by
  induction qs generalizing s with
  | nil => rfl
  | cons q rest ih =>
    rw [Runtime.compose,ih (fun p hp => valid p (by simp [hp]))]
    change execute p r y rest (Runtime.block (operations p) r "X" "Y" q s)
        (project y memory (Runtime.block (operations p) r "X" "Y" q s))=
      execute p r y rest (Runtime.block (operations p) r "X" "Y" q s)
        (scatter (TileShape.allIndices [2,1024/2]) p r y (Runtime.setup s q) Bool.true
          (scatter (TileShape.allIndices [2,1024/2]) p r y (Runtime.setup s q) Bool.false
            (project y memory s)))
    congr 1
    unfold Runtime.block Source.written Source.halfScatter
    rw [scatter_commutes p r y memory _ _ Bool.true _
      (block_bounds r y size q (valid q (by simp)) s Bool.true),
      scatter_commutes p r y memory _ _ Bool.false _
      (block_bounds r y size q (valid q (by simp)) s Bool.false)]
    rfl

theorem run_bytes (p : Primitives) (r : Request) (x y : Tensor) (input : Input x r)
    (out : Output x y) (memory : Memory) (qs : List Runtime.Program)
    (all : qs.Perm (Runtime.programs r)) (final : State Tagged)
    (ran : Runtime.run (operations p) r "X" "Y" qs (initial x y memory)=some final) :
    project y memory final=execute p r y qs (initial x y memory) memory := by
  obtain ⟨t,ht,hm⟩ := Runtime.complete (operations p) r "X" "Y" qs (initial x y memory)
  have equal : t=final := Option.some.inj (ht.symm.trans ran)
  subst t
  have projection : project y memory final=
      project y memory (Runtime.compose (operations p) r "X" "Y" qs (initial x y memory)) := by
    unfold project reference
    rw [hm]
  rw [projection,compose_commutes p r y (out.size.trans input.size) qs
    (fun q hq => ((Runtime.mem_programs r q).mp (all.mem_iff.mp hq)).1),entry_project]

theorem input_backing_frame (x y : Tensor) (out : Output x y) (memory : Memory) (values : Nat → Word)
    (a : Nat) (inside : x.actual.storage.base≤a ∧ a<x.actual.storage.base+4*x.actual.storage.elements) :
    reference y memory values a=memory a := by
  apply reference_frame
  unfold footprint
  rcases out.fresh with before|after <;> omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Bytes
