import bench.tritonbench_g.chunk_retention.Proofs.Sources.BackwardHSource

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
    ∃ t, step (engine (withDot M D)) (loopStmt c scale terminal) current=some t ∧
      Context M c initial t ∧ HasCarry c (trajectory M D c initial scale terminal c.chunks).1 t ∧
      t.mem=(written M D c initial scale terminal).mem := by
  let I := fun n t => Context M c initial t ∧
    HasCarry c (trajectory M D c initial scale terminal n).1 t ∧
    t.mem=(trajectory M D c initial scale terminal n).2.mem
  have advance : ∀ n, n<c.chunks → ∀ t, I n t → ∃ u,
      run (engine (withDot M D)) (parts c scale terminal).body
        (t.setReg "__rev_i_t" .int [] (fun _ => (1-(c.chunks : Int))+(n : Int)))=some u ∧ I (n+1) u := by
    intro n hn t reached
    let carry := (trajectory M D c initial scale terminal n).1
    let next := bodyExecution M D c scale terminal initial t n carry reached.1 reached.2.1 hn
    refine ⟨next.val,?_,bodyContext M D c scale terminal initial t n carry reached.1 reached.2.1 hn,?_,?_⟩
    · simpa [indexState,add_comm,add_left_comm,add_assoc,sub_eq_add_neg] using next.property
    · exact bodyCarry M D c scale terminal initial t n carry reached.1 reached.2.1 hn
    · rw [bodyMemory]
      simp only [trajectory,KernelFold.sequenceValue,advance,stored,masked_scatter_memory,reached.2.2]
      rfl
  obtain ⟨t,ran,done⟩ := VeriTile.Bench.Repairs.ChunkLinearAttn.SignedRange.clock
    (engine (withDot M D)) "__rev_i_t" (parts c scale terminal).body
    (1-(c.chunks : Int)) c.chunks I advance 0 (Nat.zero_le _) current ⟨ctx,ha,hm⟩
  refine ⟨t,?_,done⟩
  have first : (intParts c scale terminal).start =
      .sub .int .nil (.constInt 0) (.sub .int .nil (.constInt (Int.ofNat c.chunks)) (.constInt 1)) := rfl
  have last : (intParts c scale terminal).stop = .constInt 1 := rfl
  have stride : (intParts c scale terminal).stride = .constNat 1 := rfl
  simp only [loopStmt,step,first,last,stride,evalOp_unfold,Option.bind,Option.bind_some,numeric,bop]
  change rangeInt (engine (withDot M D)) "__rev_i_t" (0-((c.chunks : Int)-1)) 1 1
    (intParts c scale terminal).body current=some t
  have beginEq : (0 : Int)-((c.chunks : Int)-1)=(1-(c.chunks : Int))+(0 : Nat) := by omega
  have endEq : (1-(c.chunks : Int))+(c.chunks : Int)=1 := by omega
  rw [beginEq]
  simpa only [endEq] using ran

theorem source_parts (c : Shape) (scale : ℝ) (terminal : Bool) :
    (kernel c scale terminal).surfaceBody=(parts c scale terminal).before ++
      [loopStmt c scale terminal] := rfl

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
  simpa only [Option.bind_some,run,ht,Option.bind,bind,pure] using (rfl : some t=some t)

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
