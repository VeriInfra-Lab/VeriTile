import bench.tritonbench_g.layer_norm_triton.Proofs.Real
import bench.tritonbench_g.layer_norm_triton.Proofs.Geometry

namespace VeriTile.Bench.Repairs.LayerNormTriton.RealProof
open Geometry
noncomputable section
set_option maxHeartbeats 2000000

theorem group_partition (M G : Nat) (hg : 0<G) :
    ((List.range (min G M)).flatMap (groupRows M G)).Perm (List.range M) := by
  have hd : ((List.range (min G M)).flatMap (groupRows M G)).Nodup := by
    apply List.nodup_flatMap.mpr
    refine ⟨fun _ _ => group_nodup M G _ hg,?_⟩
    apply (List.nodup_range (n:=min G M)).imp_of_mem
    intro a b ha hb hab row hra hrb
    have he1 := ((group_residue M G a row hg (List.mem_range.mp ha)).mp hra).2
    have he2 := ((group_residue M G b row hg (List.mem_range.mp hb)).mp hrb).2
    exact hab (he1.symm.trans he2)
  apply (List.perm_ext_iff_of_nodup hd (List.nodup_range)).mpr
  intro row
  simp only [List.mem_flatMap,List.mem_range]
  constructor
  · rintro ⟨group,hg',hm⟩
    exact ((group_residue M G group row hg hg').mp hm).1
  · intro hr
    obtain ⟨group,hg',_⟩ := unique_group M G row hg hr
    exact ⟨group,hg'.1,hg'.2⟩

theorem map_range_sum (f : Nat → ℝ) (n : Nat) :
    ((List.range n).map f).sum=∑ i ∈ Finset.range n,f i := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ,Finset.sum_range_succ,ih,List.sum_append]

theorem sum_flatMap (rows : List Nat) (f : Nat → List ℝ) :
    (rows.flatMap f).sum=(rows.map (fun r => (f r).sum)).sum := by
  induction rows with
  | nil => simp
  | cons r rows ih => simp [List.sum_append,ih]

theorem grouped_sum (M G : Nat) (hg : 0<G) (f : Nat → ℝ) :
    (∑ group ∈ Finset.range (min G M),((groupRows M G group).map f).sum)=
      ∑ row ∈ Finset.range M,f row := by
  have hp := ((group_partition M G hg).map f).sum_eq
  simpa only [List.map_flatMap,sum_flatMap,map_range_sum] using hp

theorem fold_add_real (values : Nat → ℝ) (rows : List Nat) (a : ℝ) :
    rows.foldl (fun acc row => acc+values row) a=a+(rows.map values).sum := by
  induction rows generalizing a with
  | nil => simp
  | cons row rows ih => simp only [List.foldl_cons,ih,List.map_cons,List.sum_cons]; ring

theorem finalLane_real (plan : FinalPlan) (groups : Nat) (partials : Nat → ℝ) (lane : Fin 32) :
    finalLane ops plan groups partials lane=
      ∑ iteration ∈ Finset.range ((groups+31)/32),
        if iteration*32+lane.val<groups then partials (iteration*32+lane.val) else 0 := by
  simp only [finalLane,ordered_add]
  change (List.range _).foldl (fun acc iteration => acc+_) 0=_
  rw [fold_add_real,zero_add,map_range_sum]
  rfl

theorem tile_sum (f : Nat → ℝ) (q K : Nat) :
    (∑ i ∈ Finset.range q,∑ j ∈ Finset.range K,f (i*K+j))=∑ j ∈ Finset.range (q*K),f j := by
  induction q with
  | zero => simp
  | succ q ih =>
    rw [Finset.sum_range_succ,ih,Nat.succ_mul,Finset.sum_range_add]

theorem final_real (plan : FinalPlan) (groups : Nat) (partials : Nat → ℝ) (column : Nat) :
    finalValue ops plan groups partials column=∑ group ∈ Finset.range groups,partials group := by
  unfold finalValue
  rw [sumTree_real,((plan.complete column).map _).sum_eq,←List.ofFn_eq_map,List.sum_ofFn]
  simp only [finalLane_real]
  rw [Fin.sum_univ_eq_sum_range (fun lane => ∑ iteration ∈ Finset.range ((groups+31)/32),
    if iteration*32+lane<groups then partials (iteration*32+lane) else 0),Finset.sum_comm]
  rw [tile_sum (fun j => if j<groups then partials j else 0)]
  apply VeriTile.Bench.TritonBenchG.LayerNormFwd.Formula.sum_padded
  have hc := Nat.div_add_mod (groups+31) 32
  have hm := Nat.mod_lt (groups+31) (by decide : 0<32)
  omega

theorem weightGradient_real (plan : Plan N B) (last : FinalPlan) (M G : Nat) (hg : 0<G)
    (x dy : Nat → Nat → ℝ) (stats : Nat → Reuse.Statistics ℝ) (col : Nat) (hc : col<N) :
    finalValue ops last (min G M) (fun group => (groupValue ops plan M G group x dy stats col).1) col=
      Spec.weightGradient N M x dy stats col := by
  rw [final_real]
  simp only [group_real]
  rw [grouped_sum M G hg]
  simp [Spec.weightGradient,VeriTile.Bench.Repairs.FastLayernorm.RealProof.weightGradientSpec,hc]

theorem biasGradient_real (plan : Plan N B) (last : FinalPlan) (M G : Nat) (hg : 0<G)
    (x dy : Nat → Nat → ℝ) (stats : Nat → Reuse.Statistics ℝ) (col : Nat) (hc : col<N) :
    finalValue ops last (min G M) (fun group => (groupValue ops plan M G group x dy stats col).2) col=
      Spec.biasGradient N M dy col := by
  rw [final_real]
  simp only [group_real]
  rw [grouped_sum M G hg]
  simp [Spec.biasGradient,VeriTile.Bench.Repairs.FastLayernorm.RealProof.biasGradientSpec,hc]

#axiomsClean group_partition
#axiomsClean final_real
#axiomsClean weightGradient_real
#axiomsClean biasGradient_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.RealProof
