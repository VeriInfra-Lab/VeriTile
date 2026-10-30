import bench.tritonbench_g.rbe_triton_transform.Proofs.RuntimeModel
import bench.tritonbench_g.rbe_triton_transform.Proofs.Selected

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Runtime
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

/-- Each program executes the whole selected DSL against the memory left by
its predecessors. Neither the runtime nor its premises supply an output. -/
def run (ops : Operations α) (r : Request) (X Y : RegionName) : List Program → State α → Option (State α)
  | [],s => some s
  | q::rest,s => (FP.Structural.exec ops.backend (Selected.source (parameters r X Y)) (setup s q)).bind
      (run ops r X Y rest)

theorem complete (ops : Operations α) (r : Request) (X Y : RegionName)
    (qs : List Program) (s : State α) :
    ∃ t,run ops r X Y qs s=some t ∧ t.mem=(compose ops r X Y qs s).mem := by
  induction qs generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons q rest ih =>
    obtain ⟨t,ht,hm⟩ := Selected.complete ops (parameters r X Y) (setup s q)
    obtain ⟨u,hu,hv⟩ := ih t
    refine ⟨u,by simpa [run,ht] using hu,?_⟩
    exact hv.trans (compose_congr ops r X Y rest t _ hm)

/-- Every output is produced by the same composed successful execution. Empty
requests execute no programs. Arbitrary scheduling permutations are covered. -/
theorem whole (ops : Operations α) (r : Request) (X Y : RegionName)
    (qs : List Program) (all : qs.Perm (programs r)) (s : State α) (different : X≠Y) :
    ∃ final,run ops r X Y qs s=some final ∧
      (∀ b m c,b<r.batches → m<r.rows → c<r.columns →
        (final.mem Y (r.offset b m c)).read .fp32=
          scalarResult ops r (fun o => (s.mem X o).read .fp32) b m c) ∧
      (∀ region o,region≠Y ∨ r.size≤o → final.mem region o=s.mem region o) := by
  obtain ⟨final,ran,hm⟩ := complete ops r X Y qs s
  have valid : ∀ q∈qs,q.1<r.batches := fun q hq => ((mem_programs r q).mp (all.mem_iff.mp hq)).1
  refine ⟨final,ran,?_,?_⟩
  · intro b m c hb hr hc
    rw [hm]
    apply compose_value ops r X Y qs s _ (fun _ => rfl) b m c hr hc valid different
    obtain ⟨l,hl,ha,hb',hr',hc'⟩ := coverage r b m c hb hr hc
    refine ⟨(batch l,program l),all.mem_iff.mpr ((mem_programs r _).mpr ((mem_schedule r l).mp hl)),?_,?_,?_,?_⟩
    · exact lane l
    · exact imaginary l
    · exact ha
    · change r.offset b m c=r.offset (batch l) (row r l) (column r l)
      rw [hb',hr',hc']
  · intro region o outside
    rw [hm]
    exact compose_frame ops r X Y qs s valid region o outside

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Runtime
