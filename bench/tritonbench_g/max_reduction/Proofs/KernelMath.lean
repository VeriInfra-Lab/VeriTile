import bench.tritonbench_g.max_reduction.Proofs.SourceGrid

/-! Kernel-only attained-maximum and least-index connection. These helpers are
shared with the optional historical whole-grid results in ASTCorrect.lean. -/
namespace VeriTile.Bench.Repairs.MaxReduction.KernelMath
open VeriTile.Triton FP.Structural SourceGrid
noncomputable section
open Classical
set_option maxHeartbeats 3000000

def engine (algebra : Compute.Algebra A) (tree : Tree) : Source.Engine A :=
  Source.treeEngine algebra (fun _ => tree)

theorem dimension_target (algebra : Compute.Algebra A) (N K BN j : Nat)
    (plan : ReductionPlan BN) (input : Nat → A) :
    target (engine algebra plan.tree) input N K BN j=
      plan.tree.eval algebra.indexedCombine
        (fun n => (if n<N then input (j/K*N*K+n*K+j%K) else algebra.padding,n)) := by
  apply Compute.tree_congr
  intro n hn
  simp [(plan.complete n).mp hn,engine,Source.treeEngine]

theorem target_local (e : Source.Engine A) (M N K BN j : Nat) (hj : j<M*K)
    (f g : Nat → A) (same : ∀ i,i<M*N*K → f i=g i) :
    target e f N K BN j=target e g N K BN j := by
  have hk : 0<K := by nlinarith
  have hm := (Nat.div_lt_iff_lt_mul hk).mpr hj
  unfold target
  apply congrArg (e.indexed BN)
  funext n
  split_ifs with hn
  · exact same _ (Address.input_bound M N K (j/K) n.val (j%K) hm hn (Nat.mod_lt j hk))
  · rfl

theorem target_real (input : Nat → ℝ) (N K BN j : Nat) (hn : 0<N) (covers : N≤BN)
    (plan : ReductionPlan BN) :
    ∃ (answer : ℝ) (index : Nat),
      target (engine Compute.real plan.tree) (fun i => (input i : WithBot ℝ)) N K BN j=((answer : WithBot ℝ),index) ∧
      RealModel.IsLeastArgmax (fun n => input (j/K*N*K+n*K+j%K)) N answer index := by
  obtain ⟨answer,index,he,hs⟩ := RealModel.indexed_correct (fun n => input (j/K*N*K+n*K+j%K)) N BN hn covers plan
  have observed := Compute.observe_tree plan.tree
    (fun n => (if n<N then (input (j/K*N*K+n*K+j%K) : WithBot ℝ) else ⊥,n))
  have values : (fun n => Compute.observe (if n<N then (input (j/K*N*K+n*K+j%K) : WithBot ℝ) else ⊥,n))=
      (fun n => if n<N then some (input (j/K*N*K+n*K+j%K),n) else none) := by
    funext n
    split_ifs <;> rfl
  rw [values] at observed
  change _=RealModel.indexed plan.tree (fun n => input (j/K*N*K+n*K+j%K)) N at observed
  rw [he] at observed
  have ht := dimension_target Compute.real N K BN j plan (fun i => (input i : WithBot ℝ))
  simp only [Compute.real] at ht
  rw [←ht] at observed
  let pair := target (engine Compute.real plan.tree) (fun i => (input i : WithBot ℝ)) N K BN j
  have hp : Compute.observe pair=some (answer,index) := observed
  have eq : pair=((answer : WithBot ℝ),index) := by
    rcases pair with ⟨value,idx⟩
    cases value with
    | bot => simp [Compute.observe] at hp
    | coe value => simpa [Compute.observe] using hp
  exact ⟨answer,index,eq,hs⟩

/-- One actual indexed-reduction program has an attained maximum and the
least original index for each active row, plus an exact exterior cell frame. -/
theorem dimension_rows (M N K BM BN p k : Nat) (hN : 0 < N) (covers : N ≤ BN)
    (plan : ReductionPlan BN) (hk : k < K) (input : Nat → ℝ)
    (s : State (WithBot ℝ))
    (binding : ∀ j, j < M*N*K → (s.mem "input" j).read .real = (input j : WithBot ℝ)) :
    ∃ t,
      FP.Structural.exec (engine Compute.real plan.tree).algebra
        (dimension "input" "values" "indices" M N K BM BN) (positioned s p k) = some t ∧
      (∀ i : Fin BM, p*BM+i.val < M → ∃ (answer : ℝ) (index : Nat),
        t.mem "values" ((p*BM+i.val)*K+k) = .mk .real (answer : WithBot ℝ) ∧
        t.mem "indices" ((p*BM+i.val)*K+k) = .mk .int (index : Int) ∧
        RealModel.IsLeastArgmax (fun n => input ((p*BM+i.val)*N*K+n*K+k)) N answer index) ∧
      (∀ (r : RegionName) (o : Nat),
        (r ≠ "values" ∧ r ≠ "indices") ∨
          (∀ i : Fin BM, p*BM+i.val < M → o ≠ (p*BM+i.val)*K+k) →
        t.mem r o = s.mem r o) := by
  have hn : 0 < BN := hN.trans_le covers
  obtain ⟨t,he,hm⟩ := dimension_program (engine Compute.real plan.tree)
    M N K BM BN (p,k) hk hn s
  refine ⟨t,he,?_,?_⟩
  · intro i hi
    let j := (p*BM+i.val)*K+k
    have hj : j < M*K := Address.output_bound M K _ _ hi hk
    have member (flag : Bool) : (flag,j) ∈ events M K BM (p,k) := by
      have job : j ∈ rowJobs M K BM (p,k) := by
        apply List.mem_map.mpr
        refine ⟨(i,PUnit.unit),?_,rfl⟩
        exact List.mem_filter.mpr ⟨TileShape.mem_allIndices _ _,by simpa using hi⟩
      cases flag <;> simp [events,job]
    have hv := VeriTile.Bench.Repairs.ApplyPenalty.Scatter.readback_injective
      (events M K BM (p,k)) key
      (cell (target (engine Compute.real plan.tree)
        (fun j => (s.mem "input" j).read .real) N K BN)) (flatten s)
      (Bool.false,j) (member Bool.false) key_injective
    have hx := VeriTile.Bench.Repairs.ApplyPenalty.Scatter.readback_injective
      (events M K BM (p,k)) key
      (cell (target (engine Compute.real plan.tree)
        (fun j => (s.mem "input" j).read .real) N K BN)) (flatten s)
      (Bool.true,j) (member Bool.true) key_injective
    rw [←hm] at hv hx
    have localInput := target_local (engine Compute.real plan.tree) M N K BN j hj
      (fun j => (s.mem "input" j).read .real) (fun j => (input j : WithBot ℝ)) binding
    obtain ⟨answer,index,result,math⟩ := target_real input N K BN j hN covers plan
    have values : target (engine Compute.real plan.tree)
        (fun j => (s.mem "input" j).read .real) N K BN j = ((answer : WithBot ℝ),index) :=
      localInput.trans result
    have divj : j/K = p*BM+i.val := by
      dsimp [j]
      rw [Nat.mul_comm _ K,Nat.mul_add_div (Nat.zero_lt_of_lt hk),Nat.div_eq_of_lt hk,Nat.add_zero]
    have modj : j%K = k := by simp [j,Nat.mod_eq_of_lt hk]
    refine ⟨answer,index,?_,?_,?_⟩
    · simpa [flatten,key,cell,values] using hv
    · simpa [flatten,key,cell,values] using hx
    · simpa [divj,modj] using math
  · intro r o outside
    change flatten t (r,o) = flatten s (r,o)
    rw [hm]
    apply VeriTile.Bench.Repairs.ApplyPenalty.Scatter.frame
    rintro ⟨flag,j⟩ member eq
    have job : j ∈ rowJobs M K BM (p,k) := by
      cases flag <;> simpa [events] using member
    obtain ⟨i,hi,jobEq⟩ := List.mem_map.mp job
    have active : p*BM+i.1.val < M := of_decide_eq_true (List.mem_filter.mp hi).2
    have off : j = o := (congrArg Prod.snd eq).symm
    rcases outside with regions | missed
    · have region := congrArg Prod.fst eq
      cases flag <;> simp [key] at region <;> tauto
    · exact missed i.1 active (off.symm.trans jobEq.symm)

#axiomsClean dimension_rows

end
end VeriTile.Bench.Repairs.MaxReduction.KernelMath
