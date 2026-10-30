import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.API
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.RealForward

namespace VeriTile.Bench.Repairs.FastCeLoss.RealProof
open Source Complete API Finset
noncomputable section
set_option maxHeartbeats 4000000

theorem logical_count (s : Shape) (labels : Nat → Int) (logical : Nat → Option (Fin s.vocab))
    (codes : ∀ row<s.rows,labels row=labelCode (logical row)) :
    Source.count s labels=Math.count (fun row : Fin s.rows => logical row.val) := by
  unfold Source.count Math.count
  symm
  apply Finset.card_bij (fun (r : Fin s.rows) _ => r.val)
  · intro r hr
    refine mem_filter.mpr ⟨mem_range.mpr r.isLt,?_⟩
    have present := (mem_filter.mp hr).2
    rw [codes r.val r.isLt]
    cases h : logical r.val with
    | none => simp [h] at present
    | some k => simp only [labelCode]; omega
  · intro a _ b _ h
    exact Fin.ext h
  · intro r hr
    have bound := mem_range.mp (mem_filter.mp hr).1
    refine ⟨⟨r,bound⟩,?_,rfl⟩
    apply mem_filter.mpr
    refine ⟨mem_univ _,?_⟩
    have present := (mem_filter.mp hr).2
    rw [codes r bound] at present
    cases h : logical r with
    | none => simp [h,labelCode] at present
    | some k => simp [h]

theorem row_upstream_real (s : Shape) (labels : Nat → Int) (items : Option ℝ) (u : ℝ) :
    rowUpstream arithmetic s labels (items.map (fun x => (some x : WithBot ℝ))) (some u)=
      some (u/items.getD (Source.count s labels : ℝ)) := by
  unfold rowUpstream
  rw [denominator_real]
  rfl

theorem gradient_graph_real (o : Options) (s : Shape) (b : Buffers) (labels : Nat → Int)
    (entry : Memory (WithBot ℝ)) (layout : ForwardLayout s b) (x : Nat → Nat → ℝ)
    (logical : Nat → Option (Fin s.vocab)) (items : Option ℝ) (u : ℝ)
    (codes : ∀ row<s.rows, labels row=labelCode (logical row))
    (inputs : ∀ row<s.rows, ∀ col<s.vocab, entry (addr b.logits (row*s.vocab+col))=some (x row col))
    (row : Fin s.rows) (col : Fin s.vocab) :
    gradientGraph arithmetic o s b labels (items.map (fun x => (some x : WithBot ℝ))) (some u)
      entry row.val col.val=
      some (Math.gradient o.softcap o.scaling
        (Math.denominator (fun row : Fin s.rows => logical row.val) items) u
        (fun row : Fin s.rows => logical row.val) (fun row col => x row.val col.val) row col) := by
  unfold gradientGraph
  rw [inputs row.val row.isLt col.val col.isLt,
    lse_graph_real o s b entry layout row.val (x row.val) (inputs row.val row.isLt),
    row_upstream_real,backward_scalar,codes row.val row.isLt,
    logical_count s labels logical codes]
  cases h : logical row.val with
  | none => simp [scalarGradient,labelCode,Math.gradient,h]
  | some k =>
    have valid : (k.val : Int)≠-100 := by omega
    have same : (col.val : Int)=(k.val : Int) ↔ col=k := by simp [Fin.ext_iff]
    simp [scalarGradient,labelCode,valid,same,Math.gradient,Math.denominator,h]

/-- Independent Real whole loss: the actual source host result cell is connected
to the logical vocabulary formula, including the input-derived ignore count. -/
theorem loss_output (o : Options) (s : Shape) (b : Buffers) (labels : Nat → Int)
    (entry : Memory (WithBot ℝ)) (layout : ForwardLayout s b) (x : Nat → Nat → ℝ)
    (logical : Nat → Option (Fin s.vocab)) (items : Option ℝ)
    (codes : ∀ row<s.rows, labels row=labelCode (logical row))
    (inputs : ∀ row<s.rows, ∀ col<s.vocab, entry (addr b.logits (row*s.vocab+col))=some (x row col)) :
    (API.forward arithmetic o s b labels (items.map (fun x => (some x : WithBot ℝ))) entry).memory
      (addr b.scalar 0)=
      some (Math.loss o.softcap o.scaling
        (Math.denominator (fun row : Fin s.rows => logical row.val) items)
        (fun row : Fin s.rows => logical row.val) (fun row col => x row.val col.val)) := by
  rw [API.mean_output arithmetic o s b labels _ entry layout,
    scalar_graph_real o s b labels entry layout x logical items codes inputs,
    logical_count s labels logical codes]
  rfl

/-- Same forward entry and same saved LSE; the mean autograd chain produces the
upstream read by the in-place custom backward. There is no saved-LSE premise. -/
theorem gradient_output (o : Options) (s : Shape) (b : Buffers) (labels : Nat → Int)
    (entry : Memory (WithBot ℝ)) (layout : ForwardLayout s b) (fresh : PrivateBuffers b)
    (x : Nat → Nat → ℝ) (logical : Nat → Option (Fin s.vocab)) (items : Option ℝ) (u : ℝ)
    (codes : ∀ row<s.rows, labels row=labelCode (logical row))
    (inputs : ∀ row<s.rows, ∀ col<s.vocab, entry (addr b.logits (row*s.vocab+col))=some (x row col))
    (row : Fin s.rows) (col : Fin s.vocab) :
    (API.backward arithmetic o s b labels (items.map (fun x => (some x : WithBot ℝ))) (some u) entry).memory
      (addr b.logits (row.val*s.vocab+col.val))=
      some (Math.gradient o.softcap o.scaling
        (Math.denominator (fun row : Fin s.rows => logical row.val) items) u
        (fun row : Fin s.rows => logical row.val) (fun row col => x row.val col.val) row col) := by
  rw [API.gradient_output arithmetic o s b labels _ (some u) entry layout fresh row.val col.val row.isLt col.isLt]
  exact gradient_graph_real o s b labels entry layout x logical items u codes inputs row col

#axiomsClean logical_count
#axiomsClean gradient_graph_real
#axiomsClean loss_output
#axiomsClean gradient_output
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.RealProof
