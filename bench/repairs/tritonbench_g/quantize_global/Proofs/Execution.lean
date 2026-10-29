import bench.repairs.tritonbench_g.quantize_global.Proofs.Source
import bench.repairs.tritonbench_g.quantize_global.Proofs.Indices
namespace QuantizeGlobalRepair
open VeriTile.Triton FP.Structural QuantizeIntegerStage
noncomputable section
set_option maxHeartbeats 2000000

structure Parameters where
  X : RegionName
  Maximum : RegionName
  Out : RegionName
  n : Nat

def Parameters.Separated (p : Parameters) : Prop := p.X ≠ p.Out ∧ p.Maximum ≠ p.Out

def Inputs [Inhabited α] (p : Parameters) (xs : Nat → α) (m : α) (s : State α) : Prop :=
  (∀ i, i < p.n → (s.mem p.X i).read .real = xs i) ∧ (s.mem p.Maximum 0).read .real = m

def patch (p : Parameters) (values : Nat → BitVec 8) (indices : List Nat)
    (mem : RegionName → Nat → Cell α) : RegionName → Nat → Cell α :=
  fun r i => if r = p.Out ∧ i < p.n ∧ i ∈ indices then .mk .int (values i).toInt else mem r i

def complete (p : Parameters) (values : Nat → BitVec 8)
    (mem : RegionName → Nat → Cell α) : RegionName → Nat → Cell α :=
  fun r i => if r = p.Out ∧ i < p.n then .mk .int (values i).toInt else mem r i

def atProgram (s : State α) (v : Invocation) : State α :=
  { s with pids := fun ax => if ax = 0 then v.pid else 0 }

def executeProgram [Inhabited α] (rest : Algebra α) (a : Scalar α)
    (p : Parameters) (v : Invocation) (s : State α) : Option (State α) :=
  QuantizeIntegerStage.exec (backend rest a) (some a.round64)
    (kernel p.X p.Maximum p.Out p.n v.base v.config.block) (atProgram s v)

def execute [Inhabited α] (rest : Algebra α) (a : Scalar α)
    (p : Parameters) : List Invocation → State α → Option (State α)
  | [], s => some s
  | v::vs, s => (executeProgram rest a p v s).bind (execute rest a p vs)

theorem offset_injective (s : State α) (base B : Nat) : Function.Injective (offset s base B) := by
  intro i j h
  have hf : i.1 = j.1 := Fin.ext (by dsimp [offset] at h; omega)
  exact Prod.ext hf (Subsingleton.elim _ _)

theorem offset_program (s : State α) (v : Invocation) (i : TileIndex [v.config.block]) :
    offset (atProgram s v) v.base v.config.block i = v.offset i.1.val := by
  simp [offset,atProgram,Invocation.offset,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]

theorem program_result [Inhabited α] (rest : Algebra α) (a : Scalar α)
    (p : Parameters) (xs : Nat → α) (m : α) (s : State α) (v : Invocation)
    (inputs : Inputs p xs m s) :
    ∃ t, executeProgram rest a p v s = some t ∧
      t.mem = patch p (fun i => scalar a (xs i) m) v.indices s.mem := by
  obtain ⟨t, ht, hm⟩ := source_run rest a (atProgram s v) p.X p.Maximum p.Out p.n v.base v.config.block
  refine ⟨t,ht,hm.trans ?_⟩
  funext r i
  by_cases h : r = p.Out ∧ i < p.n ∧ i ∈ v.indices
  · obtain ⟨hr,hi,hmem⟩ := h
    obtain ⟨lane,hl,he⟩ := List.mem_map.mp hmem
    have hb : lane < v.config.block := List.mem_range.mp hl
    let j : TileIndex [v.config.block] := (⟨lane,hb⟩,PUnit.unit)
    have ho : offset (atProgram s v) v.base v.config.block j = i := by rw [offset_program]; exact he
    have readback := State.masked_scatter_readback p.Out
      (offset (atProgram s v) v.base v.config.block)
      (fun k => Cell.mk .int (word a (atProgram s v) p.X p.Maximum v.base v.config.block k).toInt)
      (fun k => offset (atProgram s v) v.base v.config.block k < p.n)
      (atProgram s v) j (by simpa [ho] using hi)
      (offset_injective (atProgram s v) v.base v.config.block)
    change (written a (atProgram s v) p.X p.Maximum p.Out p.n v.base v.config.block).mem _ _ = _ at readback
    rw [ho] at readback
    subst r
    rw [readback]
    simp only [word,ho]
    simp [patch,hi,hmem,atProgram,inputs.1 i hi,inputs.2]
  · have frame := State.masked_scatter_frame p.Out
      (offset (atProgram s v) v.base v.config.block)
      (fun k => Cell.mk .int (word a (atProgram s v) p.X p.Maximum v.base v.config.block k).toInt)
      (fun k => offset (atProgram s v) v.base v.config.block k < p.n)
      (TileShape.allIndices [v.config.block]) r i ?_ (atProgram s v)
    · simpa [written,patch,h,atProgram] using frame
    · by_cases hr : r = p.Out
      · right
        intro j _ hj he
        apply h
        refine ⟨hr,by simpa [he] using hj,?_⟩
        apply List.mem_map.mpr
        exact ⟨j.1.val,List.mem_range.mpr j.1.isLt,(offset_program s v j).symm.trans he.symm⟩
      · exact Or.inl hr

theorem patch_inputs [Inhabited α] (p : Parameters) (values : Nat → BitVec 8)
    (xs : Nat → α) (m : α) (s t : State α) (indices : List Nat)
    (sep : p.Separated) (inputs : Inputs p xs m s)
    (ht : t.mem = patch p values indices s.mem) : Inputs p xs m t := by
  constructor
  · intro i hi; rw [ht]; simpa [patch,sep.1] using inputs.1 i hi
  · rw [ht]; simpa [patch,sep.2] using inputs.2

theorem patch_append (p : Parameters) (values : Nat → BitVec 8)
    (left right : List Nat) (mem : RegionName → Nat → Cell α) :
    patch p values right (patch p values left mem) = patch p values (left++right) mem := by
  funext r i
  by_cases hr : r = p.Out <;> by_cases hi : i < p.n <;>
    by_cases hl : i ∈ left <;> by_cases hu : i ∈ right <;>
    simp [patch,hr,hi,hl,hu]

theorem execution_result [Inhabited α] (rest : Algebra α) (a : Scalar α)
    (p : Parameters) (xs : Nat → α) (m : α) (vs : List Invocation)
    (sep : p.Separated) (s : State α) (inputs : Inputs p xs m s) :
    ∃ t, execute rest a p vs s = some t ∧
      t.mem = patch p (fun i => scalar a (xs i) m) (footprint vs) s.mem := by
  induction vs generalizing s with
  | nil => exact ⟨s,rfl,by funext r i; simp [patch,footprint]⟩
  | cons v vs ih =>
    obtain ⟨u,hu,hum⟩ := program_result rest a p xs m s v inputs
    obtain ⟨t,ht,htm⟩ := ih u (patch_inputs p _ xs m s u v.indices sep inputs hum)
    refine ⟨t,by simp [execute,hu,ht],?_⟩
    rw [htm,hum,patch_append]
    rfl

theorem whole_grid [Inhabited α] (rest : Algebra α) (a : Scalar α)
    (p : Parameters) (xs : Nat → α) (m : α) (trials : Nat → List Config)
    (nonempty : ∀ k, k < segmentCount p.n → trials k ≠ [])
    (sep : p.Separated) (s : State α) (inputs : Inputs p xs m s) :
    ∃ t, execute rest a p (schedule p.n trials) s = some t ∧
      t.mem = complete p (fun i => scalar a (xs i) m) s.mem := by
  obtain ⟨t,ht,hm⟩ := execution_result rest a p xs m _ sep s inputs
  refine ⟨t,ht,hm.trans ?_⟩
  funext r i
  by_cases hi : i < p.n
  · simp [patch,complete,hi,schedule_covers p.n trials nonempty i hi]
  · simp [patch,complete,hi]

#specNonCircular complete avoiding [execute,executeProgram,source,kernel]
#axiomsClean whole_grid
#axiomsClean program_result
end
end QuantizeGlobalRepair
