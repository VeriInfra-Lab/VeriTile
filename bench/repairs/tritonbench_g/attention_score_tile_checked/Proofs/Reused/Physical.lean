import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPStorage

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Physical
open VeriTile Triton FP.Structural Geometry FPEngine FPStorage
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

def tracked (r : RegionName) : Prop := r="q" ∨ r="k" ∨ r="m" ∨ r="out"
def size (c : Shape) (r : RegionName) : Nat :=
  if r="q" then c.querySize else if r="k" then c.keySize else if r="m" then c.normalizerSize
  else if r="out" then c.outputSize else 0

/-- Ordinary finite normalized FP32 storage contract. Input buffers may overlap
one another; the freshly allocated returned output is separate from each. -/
structure ABI (c : Shape) where
  alloc : RegionName → Allocation
  capacity : Nat
  lengths : ∀ r, tracked r → (alloc r).length=size c r
  finite : ∀ r, tracked r → (alloc r).base+(alloc r).length≤capacity
  fresh : ∀ r, tracked r → r≠"out" → (alloc "out").Separate (alloc r)

def initial (abi : ABI c) (physical : Nat → α) : State α :=
  state (project abi.alloc physical) (fun _ => 0)
    (fun dim => if dim=0 then c.programs else if dim=1 then c.Z*c.H else 1)

def execute (M : Algebra α) (dot : Dot α) (c : Shape) (abi : ABI c) (physical : Nat → α)
    (offset size : Int) (scale : ℝ) (window complement : Bool) : Option (Nat → α) :=
  (FPReturned.hostRun M dot c (initial abi physical) offset size scale window complement).map
    (fun t => flush (abi.alloc "out") (exportMemory t "out") physical)

/-- End-to-end finite physical return: every output word is the actual typed
source recurrence, input projections are preserved, and all other physical
addresses retain their old contents. No final-output or successful-run premise. -/
theorem returned (M : Algebra α) (dot : Dot α) (c : Shape) (abi : ABI c) (physical : Nat → α)
    (offset size : Int) (scale : ℝ) (window complement : Bool) :
    ∃ result, execute M dot c abi physical offset size scale window complement=some result ∧
      (∀ (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK),
        result ((abi.alloc "out").base+c.outputBase z.val h.val+j.val)=
          FPReturned.returnedValue M dot c (initial abi physical) offset size scale window complement z h j) ∧
      (∀ (r : RegionName), tracked r → r≠"out" → ∀ i, i<(abi.alloc r).length →
        result ((abi.alloc r).base+i)=physical ((abi.alloc r).base+i)) ∧
      (∀ a, ¬(abi.alloc "out").Contains a → result a=physical a) := by
  obtain ⟨t,ht,_,hv,_⟩ := FPReturned.host_returned M dot c (initial abi physical) offset size scale window complement
  refine ⟨flush (abi.alloc "out") (exportMemory t "out") physical,?_,?_,?_,?_⟩
  · simp [execute,ht]
  · intro z h j
    have hlen : (abi.alloc "out").length=c.outputSize := by
      simpa [Physical.size] using abi.lengths "out" (by simp [tracked])
    have hi : c.outputBase z.val h.val+j.val<(abi.alloc "out").length := by
      rw [hlen]; exact output_bound c z h j
    rw [Nat.add_assoc,flush_readback _ _ _ _ hi]
    simp only [exportMemory,hv,payload_real]
  · intro r hr hne i hi
    exact flush_preserves_input _ _ (abi.fresh r hr hne) _ _ i hi
  · intro a ha
    exact flush_frame _ _ _ a ha

/-- Logical tensor values presented before resolve_conj/resolve_neg/contiguous.
The normalization port is solely an input-copy/read correspondence. -/
structure LogicalInputs (α : Type) (c : Shape) where
  q : Fin c.Z → Fin c.H → Fin c.N → Fin c.D → α
  k : Fin c.Z → Fin c.HK → Fin c.NK → Fin c.D → α
  m : Fin c.Z → Fin c.H → Fin c.R → α

structure Normalized (c : Shape) (abi : ABI c) (physical : Nat → α) (logical : LogicalInputs α c) : Prop where
  query : ∀ z h i d, physical ((abi.alloc "q").base+c.queryBase z.val h.val+i.val*c.D+d.val)=logical.q z h i d
  key : ∀ (z : Fin c.Z) (h : Fin c.HK) (j : Fin c.NK) (d : Fin c.D),
    physical ((abi.alloc "k").base+(z.val*c.HK+h.val)*c.NK*c.D+j.val*c.D+d.val)=logical.k z h j d
  normalizer : ∀ z h i, physical ((abi.alloc "m").base+c.normalizerBase z.val h.val+i.val)=logical.m z h i

/-- Normal completion binds all statements to the actual selected launch size.
The retry changes both blocks once, keeps the floor at sixteen, and persists. -/
structure Dispatch (c : Shape) where
  initialBlock : Nat
  supported : Supported initialBlock
  completion : Completion
  actual : c.B=selected initialBlock completion
  emptyFirst : c.N=0 ∨ c.outputSize=0 → completion=.first

def persistentBlock (dispatch : Dispatch c) : Nat := selected dispatch.initialBlock dispatch.completion

theorem persistent_selected (dispatch : Dispatch c) : persistentBlock dispatch=c.B := dispatch.actual.symm

theorem selected_positive (dispatch : Dispatch c) : 0<persistentBlock dispatch :=
  supported_positive _ (selected_supported _ dispatch.supported dispatch.completion)

#axiomsClean returned
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Physical
