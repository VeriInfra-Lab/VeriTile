import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.KvCacheCopy.KernelReal
open VeriTile.Triton Source
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem scatter_memory {I : Type} (xs : List I)
    (t : BlockState) (r : RegionName) (offset : I → Nat) (value : I → ℝ) :
    (xs.foldl (fun z i => z.writeMem r (offset i) (value i)) t).mem =
      xs.foldl (fun m i => fun r' o => if r'=r ∧ o=offset i then MemCell.real (value i) else m r' o) t.mem := by
  induction xs generalizing t with
  | nil => rfl
  | cons i xs ih => exact ih (t.writeMem r (offset i) (value i))

@[simp] theorem setReg_memory (s : BlockState) (r : RegName) (d : TileDType) (sh : TileShape)
    (v : Tile d sh) : (s.setReg r d sh v).mem=s.mem := rfl

theorem finalStep_memory (p : Parameters)
    (s t : BlockState) (q : Nat) :
    (finalStep p s q t).mem=(copyStep p s q t).mem := by
  simp only [finalStep,copyStep,scatterOne,scatter_memory,setReg_memory,preparedState]
  all_goals rfl

theorem copyStep_congr (p : Parameters)
    (s t u : BlockState) (q : Nat) (hm : t.mem=u.mem) :
    (copyStep p s q t).mem=(copyStep p s q u).mem := by
  simp only [copyStep,scatterOne,scatter_memory,hm]

theorem scatterOne_frame (p : Parameters)
    (s t : BlockState) (key : Bool) (q : Nat) (r : RegionName) (o : Nat)
    (h : r≠outputRegion key ∨ ∀ i : TileIndex [p.width],o≠(targetInt p s key q i).toNat) :
    (scatterOne p s key q t).mem r o=t.mem r o := by
  unfold scatterOne
  rcases h with other | miss
  · exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r other o t
  · by_cases same : r=outputRegion key
    · subst r
      exact BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o (fun i _ => (miss i).symm) t
    · exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r same o t

theorem copyStep_frame (p : Parameters)
    (s t : BlockState) (q : Nat) (r : RegionName) (o : Nat)
    (h : ∀ key,r≠outputRegion key ∨ ∀ i : TileIndex [p.width],o≠(targetInt p s key q i).toNat) :
    (copyStep p s q t).mem r o=t.mem r o := by
  unfold copyStep
  rw [scatterOne_frame p s _ Bool.false q r o (h Bool.false),
    scatterOne_frame p s _ Bool.true q r o (h Bool.true)]

theorem finalStep_controls (p : Parameters)
    (s t : BlockState) (q : Nat) (h : Controls p s t) : Controls p s (finalStep p s q t) := by
  rcases h with ⟨hs,hh,hb,hslot,hl⟩
  constructor <;> simp [finalStep,scatterOne,preparedState,hs,hh,hb,hslot,hl]

theorem finalStep_frozen (p : Parameters)
    (s t : BlockState) (q : Nat) (h : Frozen s t) : Frozen s (finalStep p s q t) := by
  constructor
  · funext o
    rw [finalStep_memory,copyStep_frame p s t q "key" o (by intro key;cases key <;> exact Or.inl (by decide))]
    exact congrFun h.1 o
  · funext o
    rw [finalStep_memory,copyStep_frame p s t q "value" o (by intro key;cases key <;> exact Or.inl (by decide))]
    exact congrFun h.2 o

private theorem loop_invariant {n : Nat} {code : List Stmt} {P : Nat → BlockState → Prop}
    (hstep : ∀ i s,i<n → P i s → ∃ t,
      stepStmts code (s.setReg "split_x" .nat [] (Tile.scalar i))=some t ∧ P (i+1) t)
    (start : Nat) (hstart : start≤n) (s : BlockState) (hs : P start s) :
    ∃ t,stepForLoopAux "split_x" start n code s=some t ∧ P n t := by
  have key : ∀ k i,n-i=k → i≤n → ∀ s,P i s →
      ∃ t,stepForLoopAux "split_x" i n code s=some t ∧ P n t := by
    intro k
    induction k with
    | zero =>
      intro i hk hi s hs
      have eq : i=n := by omega
      subst i
      exact ⟨s,by rw [stepForLoopAux];simp,hs⟩
    | succ k ih =>
      intro i hk hi s hs
      have hlt : i<n := by omega
      obtain ⟨t,ht,hp⟩ := hstep i s hlt hs
      obtain ⟨u,hu,hpu⟩ := ih (i+1) (by omega) (by omega) t hp
      refine ⟨u,?_,hpu⟩
      rw [stepForLoopAux]
      simp only [hlt,ite_true,ht]
      exact hu
  exact key (n-start) start rfl hstart s hs

/-- Successful full kernel execution with the initial-value snapshot copy. -/
theorem execute (p : Parameters) (s : BlockState) :
    ∃ final,exec (originalKernel p).toAlgKernel s=some final ∧
      final.mem=(copied p s (p.dims/p.width)).mem := by
  let P := fun q t => Controls p s t ∧ Frozen s t ∧ t.mem=(copied p s q).mem
  have initial : P 0 (preludeState p s) := by
    refine ⟨?_,⟨rfl,rfl⟩,rfl⟩
    constructor <;> simp [preludeState]
  have step : ∀ q t,q<p.dims/p.width → P q t →
      ∃ u,stepStmts (body p) (t.setReg "split_x" .nat [] (Tile.scalar q))=some u ∧ P (q+1) u := by
    intro q t _ ht
    refine ⟨finalStep p s q t,body_run p s t q ht.1 ht.2.1,?_,?_,?_⟩
    · exact finalStep_controls p s t q ht.1
    · exact finalStep_frozen p s t q ht.2.1
    · exact (finalStep_memory p s t q).trans (copyStep_congr p s t _ q ht.2.2)
  obtain ⟨final,he,hp⟩ := loop_invariant step 0 (Nat.zero_le _) (preludeState p s) initial
  refine ⟨final,?_,hp.2.2⟩
  change stepStmts (source p).body s=some final
  rw [decomposition,ProgramRewrite.append,prelude_run]
  simp [stepStmts,stepStmt,he]

theorem copied_frame (p : Parameters) (s : BlockState) (n : Nat) (r : RegionName) (o : Nat)
    (h : ∀ key q,q<n → r≠outputRegion key ∨
      ∀ i : TileIndex [p.width],o≠(targetInt p s key q i).toNat) :
    (copied p s n).mem r o=s.mem r o := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [copied,copyStep_frame p s _ n r o (fun key => h key n (by omega))]
    exact ih (fun key q hq => h key q (by omega))

theorem copyStep_readback (p : Parameters) (s t : BlockState) (key : Bool) (q : Nat)
    (inj : Function.Injective (fun i : TileIndex [p.width] => (targetInt p s key q i).toNat))
    (i : TileIndex [p.width]) :
    (copyStep p s q t).readMem (outputRegion key) (targetInt p s key q i).toNat=
      s.readMem (inputRegion key) (sourceOffset p s.pids key q i) := by
  cases key with
  | «false» => exact BlockState.scatter_readback_nd _ _ _ inj i
  | «true» =>
    simp only [BlockState.readMem,copyStep]
    rw [scatterOne_frame p s _ Bool.false q _ _ (Or.inl (by decide))]
    exact BlockState.scatter_readback_nd _ _ _ inj i

theorem copied_readback (p : Parameters) (s : BlockState) (n : Nat) (key : Bool)
    (q : Nat) (hq : q<n) (i : TileIndex [p.width])
    (unique : ∀ u v,u<n → v<n → ∀ j k : TileIndex [p.width],
      (targetInt p s key u j).toNat=(targetInt p s key v k).toNat → u=v ∧ j=k) :
    (copied p s n).readMem (outputRegion key) (targetInt p s key q i).toNat=
      s.readMem (inputRegion key) (sourceOffset p s.pids key q i) := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases last : q=n
    · subst q
      exact copyStep_readback p s _ key n (fun i j eq => (unique n n (by omega) (by omega) i j eq).2) i
    · have earlier : q<n := by omega
      simp only [copied,BlockState.readMem]
      rw [copyStep_frame p s _ n _ _ (by
        intro other
        by_cases same : key=other
        · subst other
          exact Or.inr (fun j eq => last ((unique n q (by omega) hq j i eq.symm).1.symm))
        · exact Or.inl (by cases key <;> cases other <;> simp_all [outputRegion]))]
      exact ih earlier (fun u v hu hv j k eq => unique u v (by omega) (by omega) j k eq)

#axiomsClean execute
#axiomsClean copied_frame
#axiomsClean copied_readback
end
end VeriTile.Bench.Repairs.KvCacheCopy.KernelReal
