import bench.repairs.tritonbench_g.attention_score.Proofs.FPReturned
/-! Adapted finite allocation/payload lemmas from the completed private Chunk GLA proof. -/
namespace VeriTile.Bench.Repairs.AttentionScore.FPStorage
open VeriTile Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

abbrev Memory (α : Type) := RegionName → Nat → α

/-- Both DSL real under the FP32 profile and explicit fp32 carry the same
physical FP32 payload. This conversion changes only the interpreter tag. -/
def payload : Cell α → α
  | .mk .real v => v
  | .mk .fp32 v => v
  | _ => default

def cell (explicit : Bool) (v : α) : Cell α :=
  if explicit then .mk .fp32 v else .mk .real v

@[simp] theorem payload_cell (explicit : Bool) (v : α) : payload (cell explicit v)=v := by
  cases explicit <;> rfl
@[simp] theorem payload_real (v : α) : payload (.mk .real v)=v := rfl
@[simp] theorem payload_fp32 (v : α) : payload (.mk .fp32 v)=v := rfl
@[simp] theorem cell_read_real (v : α) : (cell Bool.false v).read .real=v := by
  exact Cell.read_same _ _
@[simp] theorem cell_read_fp32 (v : α) : (cell Bool.true v).read .fp32=v := by
  exact Cell.read_same _ _

/-- All three normalized inputs and the output use the DSL real tag under
its explicit FP32 computation profile. -/
def importMemory (m : Memory α) : RegionName → Nat → Cell α := fun r o => cell Bool.false (m r o)
def exportMemory (s : State α) : Memory α := fun r o => payload (s.mem r o)
def state (m : Memory α) (pids numPids : Nat → Nat) : State α :=
  { mem:=importMemory m, regs:=fun _ _ _ => none, pids:=pids, numPids:=numPids,
    undef:=fun d _ _ => defaultValue d }

@[simp] theorem export_import (m : Memory α) (pids numPids : Nat → Nat) :
    exportMemory (state m pids numPids)=m := by
  funext r o
  exact payload_cell Bool.false (m r o)

theorem import_real (m : Memory α) (r : RegionName) (o : Nat) :
    (importMemory m r o).read .real=m r o := by simp [importMemory]

theorem exported_write (s : State α) (r : RegionName) (o : Nat) (d : Bool) (v : α) :
    exportMemory (s.write r o (cell d v)) r o=v := by
  simp [exportMemory,State.write_same]
theorem export_eq {s t : State α} (h : s.mem=t.mem) : exportMemory s=exportMemory t := by
  exact congrArg (fun mem => fun r o => payload (mem r o)) h

/-- Ordinary finite FP32 allocation in a flat word address space. Inputs may
overlap each other; only writable allocations need separation from reads. -/
structure Allocation where
  base : Nat
  length : Nat
def Allocation.Contains (a : Allocation) (p : Nat) : Prop := a.base ≤ p ∧ p<a.base+a.length
instance (a : Allocation) (p : Nat) : Decidable (a.Contains p) := inferInstanceAs (Decidable (_∧_))
def Allocation.Separate (a b : Allocation) : Prop := a.base+a.length ≤ b.base ∨ b.base+b.length ≤ a.base
def project (alloc : RegionName → Allocation) (physical : Nat → α) : Memory α :=
  fun r o => physical ((alloc r).base+o)
def flush (a : Allocation) (values : Nat → α) (physical : Nat → α) : Nat → α :=
  fun p => if a.Contains p then values (p-a.base) else physical p

theorem flush_readback (a : Allocation) (values physical : Nat → α) (i : Nat) (hi : i<a.length) :
    flush a values physical (a.base+i)=values i := by simp [flush,Allocation.Contains,hi]
theorem flush_frame (a : Allocation) (values physical : Nat → α) (p : Nat) (hp : ¬a.Contains p) :
    flush a values physical p=physical p := by simp [flush,hp]
theorem separate_not_contains (a b : Allocation) (h : a.Separate b) (i : Nat) (hi : i<b.length) :
    ¬a.Contains (b.base+i) := by
  unfold Allocation.Separate Allocation.Contains at *
  omega
theorem flush_preserves_input (a b : Allocation) (h : a.Separate b)
    (values physical : Nat → α) (i : Nat) (hi : i<b.length) :
    flush a values physical (b.base+i)=physical (b.base+i) :=
  flush_frame a values physical _ (separate_not_contains a b h i hi)

#axiomsClean export_import
#axiomsClean flush_readback
#axiomsClean flush_preserves_input
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScore.FPStorage
