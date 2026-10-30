import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.IOSchedule
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.Real

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.IOResults
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open SourceGraph (Expr)
open VecmatIO
noncomputable section
set_option maxHeartbeats 2000000

theorem output (profile : KernelProfile.Selector) (p : Parameters)
    (initial final : State Expr) (separate : SourceGraph.InputsSeparate p)
    (hBM : 0<p.BM) (hBN : 0<p.BN) (hM : p.BM∣p.M) (hN : p.BN∣p.N)
    (memory : final.mem=SourceGraph.gridExpectedMemory (KernelProfile.toPlans profile) p initial)
    (m : Fin p.M) (n : Fin p.N) :
    final.mem p.output (m.val*p.N+n.val)=Cell.mk .real
      (SourceGraph.tensorValue (KernelProfile.toPlans profile) p initial.mem m.val n.val (p.K/p.BK)) := by
  rw [memory,←SourceGraph.complete_initial_memory_eq_grid _ p initial separate]
  exact SourceGraph.complete_tensor_readback _ p initial hBM hBN hM hN m n

theorem frame (profile : KernelProfile.Selector) (p : Parameters) (initial final : State Expr)
    (memory : final.mem=SourceGraph.gridExpectedMemory (KernelProfile.toPlans profile) p initial)
    (r : RegionName) (address : Nat)
    (outside : ∀ (m : Fin p.M) (n : Fin p.N), r≠p.output ∨ address≠m.val*p.N+n.val) :
    final.mem r address=initial.mem r address := by
  rw [memory]
  apply SourceGraph.grid_reference_frame _ p initial (SourceGraph.gridCount p) (by rfl) r address
  by_cases hr : r=p.output
  · right
    intro m n i
    exact (outside
      ⟨m.val*p.BM+i.1.val,SourceGraph.tiled_coordinate_lt p.M p.BM m i.1⟩
      ⟨n.val*p.BN+i.2.1.val,SourceGraph.tiled_coordinate_lt p.N p.BN n i.2.1⟩).resolve_left (not_not.mpr hr)
  · exact Or.inl hr

/-- Bind the complete native operation recurrence to the logical channels.
Only addresses actually visited by the floor K loop are used. -/
theorem tensor_formula {α : Type} (plans : SourceGraph.PlanSelector) (p : Parameters)
    (encode : RegionName → Nat → α → Expr) (vals : ChannelValues α (inputs p))
    (mem : SourceGraph.Memory)
    (ha : ∀ (m : Fin p.M) (k : Fin p.K),
      (mem p.A (m.val*p.K+k.val)).read .real=
        encode p.A (m.val*p.K+k.val) (vals (0 : Fin 2) (m,k,())))
    (hb : ∀ (m : Fin p.M) (n : Fin p.N) (k : Fin p.K),
      (mem p.B (m.val*p.N*p.K+n.val*p.K+k.val)).read .real=
        encode p.B (m.val*p.N*p.K+n.val*p.K+k.val) (vals (1 : Fin 2) (m,n,k,())))
    (m : Fin p.M) (n : Fin p.N) (c : Nat) (hc : c≤p.K/p.BK) :
    SourceGraph.tensorValue plans p mem m.val n.val c=
      accumulate plans p (aValue p encode vals) (bValue p encode vals) m.val n.val c := by
  induction c with
  | zero => rfl
  | succ c ih =>
    rw [SourceGraph.tensorValue,accumulate,ih (by omega)]
    congr 1
    unfold SourceGraph.tensorBlock block
    have eq : (fun k : Fin p.BK => Expr.mul
        ((mem p.A (m.val*p.K+(c*p.BK+k.val))).read .real)
        ((mem p.B (m.val*p.N*p.K+n.val*p.K+(c*p.BK+k.val))).read .real)) =
        (fun k : Fin p.BK => Expr.mul
          (aValue p encode vals m.val (c*p.BK+k.val))
          (bValue p encode vals m.val n.val (c*p.BK+k.val))) := by
      funext k
      have hk := SourceGraph.tiled_coordinate_lt p.K p.BK ⟨c,by omega⟩ k
      rw [ha m ⟨_,hk⟩,hb m n ⟨_,hk⟩]
      simp only [aValue,bValue,m.isLt,n.isLt,hk,dif_pos]
    rw [eq]

theorem real_formula (decode : ValueProofs.Word → ℝ) (profile : KernelProfile.Selector)
    (p : Parameters) (vals : ChannelValues Expr (inputs p)) (mem : SourceGraph.Memory)
    (ha : ∀ (m : Fin p.M) (k : Fin p.K),
      (mem p.A (m.val*p.K+k.val)).read .real=vals (0 : Fin 2) (m,k,()))
    (hb : ∀ (m : Fin p.M) (n : Fin p.N) (k : Fin p.K),
      (mem p.B (m.val*p.N*p.K+n.val*p.K+k.val)).read .real=vals (1 : Fin 2) (m,n,k,()))
    (hK : p.BK∣p.K) (m : Fin p.M) (n : Fin p.N) :
    SourceGraph.realValue decode (SourceGraph.contract
      (SourceGraph.tensorValue (KernelProfile.toPlans profile) p mem m.val n.val (p.K/p.BK)))=
      realTarget decode p vals (0 : Fin 1) (m,n,()) := by
  rw [SourceGraph.selected_graph_real_correct,RealProofs.blocked_expected_eq_full decode p mem _ _ hK]
  change (∑ k : Fin p.K, _ * _) = ∑ k : Fin p.K, _ * _
  apply Finset.sum_congr rfl
  intro k _
  rw [ha m k,hb m n k]

#axiomsClean tensor_formula
#axiomsClean real_formula
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.IOResults
