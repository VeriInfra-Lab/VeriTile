import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardHSource

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardHSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 8000000

abbrev Accum (c : Shape) (α : Type) := (Index c → α) × State α
def advance (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α)
    (scale : ℝ) (n : Nat) (a : Accum c α) : Accum c α :=
  ⟨update M D c s scale n a.1,stored M c s a.2 n a.1⟩
def trajectory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α)
    (scale : ℝ) (terminal : Bool) (n : Nat) : Accum c α :=
  KernelFold.sequenceValue (advance M D c s scale) ⟨initValue M c terminal s,s⟩ n
def written (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α)
    (scale : ℝ) (terminal : Bool) : State α :=
  (trajectory M D c s scale terminal c.chunks).2

theorem loopExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (scale : ℝ) (terminal : Bool) (initial current : State α)
    (ctx : Context M c initial current) (ha : HasCarry c (initValue M c terminal initial) current)
    (hm : current.mem=initial.mem) :
    ∃ t, range (engine (withDot M D)) "j" 0 c.chunks 1 (parts c scale terminal).body current=some t ∧
      Context M c initial t ∧ HasCarry c (trajectory M D c initial scale terminal c.chunks).1 t ∧
      t.mem=(written M D c initial scale terminal).mem := by
  let I := fun (_ : Nat) (a : Accum c α) (s : State α) =>
    Context M c initial s ∧ HasCarry c a.1 s ∧ s.mem=a.2.mem
  have hloop := range_values (engine (withDot M D)) "j" (parts c scale terminal).body
    (advance M D c initial scale) I (fun a s => Context M c initial s ∧ HasCarry c a.1 s ∧ s.mem=a.2.mem)
    0 c.chunks 1 ⟨initValue M c terminal initial,initial⟩ current
    (by
      intro n a s h hn
      let next := bodyExecution M D c scale terminal initial s n a.1 h.1 h.2.1
      refine ⟨next.val,next.property,bodyContext M D c scale terminal initial s n a.1 h.1 h.2.1,
        bodyCarry M D c scale terminal initial s n a.1 h.1 h.2.1,?_⟩
      rw [bodyMemory]
      simp only [advance,stored,masked_scatter_memory,h.2.2])
    (by intro n a s h; exact h) ⟨ctx,ha,hm⟩
  simpa only [KernelFold.from_zero,trajectory,written] using hloop

theorem source_parts (c : Shape) (scale : ℝ) (terminal : Bool) :
    (kernel c scale terminal).surfaceBody=(parts c scale terminal).before ++
      [.forRange "j" 0 c.chunks 1 (parts c scale terminal).body] := rfl

theorem source_memory (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (scale : ℝ) (terminal : Bool) (s : State α) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (kernel c scale terminal) s=some t ∧
      t.mem=(written M D c s scale terminal).mem := by
  let start := entryExecution M D c scale terminal s
  have hc := entryContext M D c scale terminal s
  have hm : start.val.mem=s.mem := by cases terminal <;> simp [start,entryExecution]
  obtain ⟨t,ht,_,_,hmem⟩ := loopExecution M D c scale terminal s start.val hc.1 hc.2 hm
  refine ⟨t,?_,hmem⟩
  change run (engine (withDot M D)) (kernel c scale terminal).surfaceBody s=some t
  rw [source_parts,run_append,start.property]
  simpa only [Option.bind_some,run,step,ht,Option.bind,bind,pure] using (rfl : some t=some t)

theorem stored_read (M : Algebra α) (c : Shape) (initial current : State α)
    (n : Nat) (carry : Index c → α) (i : Index c) (hi : active c initial i) :
    (stored M c initial current n carry).mem "dh" (historyAddress c initial (chunk c n) i)=
      .mk .real (M.cast (some .fp32) .real .real (carry i)) := by
  apply masked_readback_active _ _ _ _ _ i hi
  intro a b ha hb he
  exact (ForwardHSource.history_injective c initial _ _ a b ha hb he).2

theorem stored_frame (M : Algebra α) (c : Shape) (initial current : State α)
    (n : Nat) (carry : Index c → α) (r : RegionName) (off : Nat)
    (h : r≠"dh" ∨ ∀ i, active c initial i → off≠historyAddress c initial (chunk c n) i) :
    (stored M c initial current n carry).mem r off=current.mem r off := by
  apply State.masked_scatter_frame
  rcases h with hr|ho
  · exact Or.inl hr
  · exact Or.inr (fun i _ hi => ho i hi)

theorem trajectory_history (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (s : State α) (scale : ℝ) (terminal : Bool) (n k : Nat) (i : Index c)
    (hn : n≤c.chunks) (hk : k<n) (hi : active c s i) :
    (trajectory M D c s scale terminal n).2.mem "dh" (historyAddress c s (chunk c k) i)=
      .mk .real (M.cast (some .fp32) .real .real ((trajectory M D c s scale terminal k).1 i)) := by
  induction n with
  | zero => omega
  | succ n ih =>
    change (stored M c s (trajectory M D c s scale terminal n).2 n
      (trajectory M D c s scale terminal n).1).mem "dh" (historyAddress c s (chunk c k) i)=_
    by_cases he : k=n
    · subst k
      exact stored_read M c s _ n _ i hi
    · rw [stored_frame M c s _ n _ "dh" _ (Or.inr (by
        intro j hj eq
        have heq := (ForwardHSource.history_injective c s (chunk c k) (chunk c n) i j hi hj eq).1
        dsimp only [chunk] at heq
        omega))]
      exact ih (by omega) (by omega)

theorem history_readback (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (s : State α) (scale : ℝ) (terminal : Bool) (k : Nat) (i : Index c)
    (hk : k<c.chunks) (hi : active c s i) :
    (written M D c s scale terminal).mem "dh" (historyAddress c s (chunk c k) i)=
      .mk .real (M.cast (some .fp32) .real .real ((trajectory M D c s scale terminal k).1 i)) :=
  trajectory_history M D c s scale terminal c.chunks k i le_rfl hk hi

theorem trajectory_frame (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (s : State α) (scale : ℝ) (terminal : Bool) (n : Nat) (r : RegionName) (off : Nat)
    (h : r≠"dh" ∨ ∀ k, k<n → ∀ i, active c s i → off≠historyAddress c s (chunk c k) i) :
    (trajectory M D c s scale terminal n).2.mem r off=s.mem r off := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change (stored M c s (trajectory M D c s scale terminal n).2 n
      (trajectory M D c s scale terminal n).1).mem r off=_
    rw [stored_frame M c s _ n _ r off (h.imp_right (fun hh => hh n (by omega)))]
    exact ih (h.imp_right (fun hh k hk => hh k (by omega)))

theorem frame (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (s : State α) (scale : ℝ) (terminal : Bool) (r : RegionName) (off : Nat)
    (h : r≠"dh" ∨ ∀ k, k<c.chunks → ∀ i, active c s i → off≠historyAddress c s k i) :
    (written M D c s scale terminal).mem r off=s.mem r off := by
  apply trajectory_frame
  exact h.imp_right (fun hh k hk => hh (chunk c k) (by dsimp [chunk]; omega))

#specNonCircular written avoiding [run,FP.Structural.exec]
#axiomsClean source_memory
#axiomsClean history_readback
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardHSource
