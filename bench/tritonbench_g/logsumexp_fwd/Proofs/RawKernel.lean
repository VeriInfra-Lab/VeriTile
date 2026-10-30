import bench.tritonbench_g.logsumexp_fwd.Proofs.Engine
import bench.tritonbench_g.logsumexp_fwd.Proofs.KernelMath
import bench.tritonbench_g.max_reduction.Proofs.SourceGrid
import bench.tritonbench_g.max_reduction.Proofs.Machine

namespace VeriTile.Bench.Repairs.LogsumexpFwd.RawKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ApplyPenalty
open MaxReduction.SourceGrid (Flat flatten write_flat)
noncomputable section
open Classical
set_option maxHeartbeats 2000000

/-- Only the scalar launch geometry used by the quotation. -/
structure Geometry where
  rows : Nat
  columns : Nat
  block : Nat
  hasScale : Bool
  positive : 0<block

def Geometry.blocks (a : Geometry) : Nat := (a.columns+a.block-1)/a.block

theorem output_offset_bound (a : Geometry) (row segment : Nat)
    (hr : row<a.rows) (hb : segment<a.blocks) : row*a.blocks+segment<a.rows*a.blocks :=
  MaxReduction.Address.output_bound _ _ _ _ hr hb

theorem active_block (a : Geometry) (segment : Nat) (hb : segment<a.blocks) :
    segment*a.block<a.columns := MaxReduction.Machine.segment_start _ _ _ a.positive hb

def compute (engines : Nat → Engine A) (a : Geometry) (scale : A) (input : Nat → A) (j : Nat) : A :=
  blockValue (engines j) input scale (j/a.blocks) (j%a.blocks) a.columns a.block a.hasScale

def programState (a : Geometry) (scale : A) (j : Nat) (s : State A) : State A :=
  (MaxReduction.SourceGrid.positioned s (j/a.blocks) (j%a.blocks)).setReg "scale" .real [] (fun _ => scale)

theorem output_index (a : Geometry) (j : Nat) :
    Source.outputAddress (j/a.blocks) (j%a.blocks) a.columns a.block=j := by
  change j/a.blocks*a.blocks+j%a.blocks=j
  simpa [Nat.mul_comm] using Nat.div_add_mod j a.blocks

/-- The complete source body writes the actual flattened scratch slot. The
scale parameter is bound to the runtime FP32 argument, not a Real literal. -/
theorem program_exec [Inhabited A] (engines : Nat → Engine A) (a : Geometry)
    (scale : A) (j : Nat) (s : State A) :
    ∃ t,FP.Structural.exec (engines j).algebra (rowBlock "input" "scratch" a.columns a.block a.hasScale)
        (programState a scale j s)=some t ∧
      flatten t=Scatter.write (flatten s) ("scratch",j)
        (.mk .real (compute engines a scale (fun i => (s.mem "input" i).read .real) j)) := by
  obtain ⟨t,he,hm⟩ := block_exec (engines j) "input" "scratch" a.columns a.block a.hasScale scale
    (programState a scale j s) a.positive (by simp [programState])
  refine ⟨t,he,?_⟩
  have flat := congrArg (fun mem => fun q : RegionName × Nat => mem q.1 q.2) hm
  change flatten t=flatten ((programState a scale j s).write "scratch" _ _) at flat
  rw [flat,write_flat]
  simp [programState,MaxReduction.SourceGrid.positioned,State.setReg,compute,output_index]
  rfl

def run [Inhabited A] (engines : Nat → Engine A) (a : Geometry) (scale : A) : List Nat → State A → Option (State A)
  | [],s => some s
  | j::rest,s => do
      let t ← FP.Structural.exec (engines j).algebra (rowBlock "input" "scratch" a.columns a.block a.hasScale)
        (programState a scale j s)
      run engines a scale rest t

theorem chain [Inhabited A] (engines : Nat → Engine A) (a : Geometry) (scale : A)
    (jobs : List Nat) (s : State A) :
    ∃ t,run engines a scale jobs s=some t ∧
      flatten t=Scatter.run jobs (fun j => (("scratch" : RegionName),j))
        (fun j => FP.Structural.Cell.mk .real (compute engines a scale (fun i => (s.mem "input" i).read .real) j))
        (flatten s) := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons j rest ih =>
    obtain ⟨mid,hm,hmemory⟩ := program_exec engines a scale j s
    obtain ⟨t,ht,htmemory⟩ := ih mid
    have input : (fun i => (mid.mem "input" i).read .real)=(fun i => (s.mem "input" i).read .real) := by
      funext i
      have frame := congrArg (fun f => f ("input",i)) hmemory
      simp [Scatter.write] at frame
      exact congrArg (FP.Structural.Cell.read .real) frame
    refine ⟨t,by simp only [run,hm]; exact ht,?_⟩
    rw [htmemory,input,hmemory]
    rfl

/-- Every row/block of the exact (N,ND) grid, with complete scratch outputs
and all input/exterior tagged cells. Empty leading shapes execute no program. -/
theorem complete [Inhabited A] (engines : Nat → Engine A) (a : Geometry) (scale : A) (s : State A) :
    ∃ t,run engines a scale (List.range (a.rows*a.blocks)) s=some t ∧
      (∀ j,j<a.rows*a.blocks → t.mem "scratch" j=.mk .real
        (compute engines a scale (fun i => (s.mem "input" i).read .real) j)) ∧
      (∀ (r : RegionName) (i : Nat), (r≠"scratch" ∨ a.rows*a.blocks ≤ i) → t.mem r i=s.mem r i) := by
  obtain ⟨t,he,hm⟩ := chain engines a scale (List.range (a.rows*a.blocks)) s
  refine ⟨t,he,?_,?_⟩
  · intro j hj
    have hv := Scatter.readback_injective (List.range (a.rows*a.blocks)) (fun j => (("scratch" : RegionName),j))
      (fun j => FP.Structural.Cell.mk .real (compute engines a scale (fun i => (s.mem "input" i).read .real) j))
      (flatten s) j (List.mem_range.mpr hj) (by intro i j h; exact (Prod.mk.inj h).2)
    rw [←hm] at hv
    exact hv
  · intro r i miss
    change flatten t (r,i)=flatten s (r,i)
    rw [hm]
    apply Scatter.frame
    intro j hj same
    have hr := congrArg Prod.fst same
    have hi := congrArg Prod.snd same
    have hj := List.mem_range.mp hj
    rcases miss with h|h
    · exact h hr
    · omega

theorem full_coverage (a : Geometry) (row block : Nat) (hr : row<a.rows) (hb : block<a.blocks) :
    row*a.blocks+block∈List.range (a.rows*a.blocks) ∧
      (row*a.blocks+block)/a.blocks=row ∧ (row*a.blocks+block)%a.blocks=block := by
  have hp : 0<a.blocks := by omega
  refine ⟨List.mem_range.mpr (output_offset_bound a row block hr hb),?_,?_⟩
  · rw [Nat.mul_comm row a.blocks,Nat.mul_add_div hp,Nat.div_eq_of_lt hb,Nat.add_zero]
  · simp [Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hb]

theorem ordered [Inhabited A] (engines : Nat → Engine A) (a : Geometry)
    (scale : A) (s : State A) (jobs : List Nat)
    (schedule : jobs.Perm (List.range (a.rows*a.blocks))) :
    ∃ t,run engines a scale jobs s=some t ∧
      (∀ j,j<a.rows*a.blocks → t.mem "scratch" j=.mk .real
        (compute engines a scale (fun i => (s.mem "input" i).read .real) j)) ∧
      (∀ (r : RegionName) i,(r ≠ "scratch" ∨ a.rows*a.blocks ≤ i) → t.mem r i=s.mem r i) := by
  obtain ⟨t,he,hm⟩ := chain engines a scale jobs s
  refine ⟨t,he,?_,?_⟩
  · intro j hj
    have member : j∈jobs := schedule.mem_iff.mpr (List.mem_range.mpr hj)
    have value := Scatter.readback_injective jobs (fun j => (("scratch" : RegionName),j))
      (fun j => FP.Structural.Cell.mk .real
        (compute engines a scale (fun i => (s.mem "input" i).read .real) j))
      (flatten s) j member (by intro i j h; exact (Prod.mk.inj h).2)
    rw [←hm] at value
    exact value
  · intro r i miss
    change flatten t (r,i)=flatten s (r,i)
    rw [hm]
    apply Scatter.frame
    intro j hj same
    have bound := List.mem_range.mp (schedule.mem_iff.mp hj)
    have region := congrArg Prod.fst same
    have index := congrArg Prod.snd same
    rcases miss with h|h
    · exact h region
    · simp only [Prod.fst,Prod.snd] at index
      omega

#axiomsClean program_exec
#axiomsClean ordered
end
end VeriTile.Bench.Repairs.LogsumexpFwd.RawKernel
