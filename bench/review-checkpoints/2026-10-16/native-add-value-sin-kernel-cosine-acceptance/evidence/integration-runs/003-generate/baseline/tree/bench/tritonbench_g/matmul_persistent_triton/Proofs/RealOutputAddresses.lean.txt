import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoopState
import bench.tritonbench_g.matmul_persistent_triton.Proofs.ScheduleInjective

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealOutputAddresses
open VeriTile.Triton RealLoopState
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000

/-- Only logical output elements must have distinct storage addresses. Stride
gaps and all masked physical tile lanes are outside this assumption. -/
def Injective (p : Parameters) : Prop :=
  Function.Injective (fun x : Fin p.M × Fin p.N => p.scm*x.1.val+p.scn*x.2.val)

theorem elements_eq (p : Parameters) (h : Injective p) (r c r' c' : Nat)
    (i j : TileIndex [p.BM,p.BN])
    (hi : RealStore.active p r c i) (hj : RealStore.active p r' c' j)
    (ho : RealStore.offset p r c i = RealStore.offset p r' c' j) :
    RealStore.rows p r i.1 = RealStore.rows p r' j.1 ∧
    RealStore.columns p c i.2.1 = RealStore.columns p c' j.2.1 := by
  have he := h (a₁ := (⟨RealStore.rows p r i.1,hi.1⟩,⟨RealStore.columns p c i.2.1,hi.2⟩))
    (a₂ := (⟨RealStore.rows p r' j.1,hj.1⟩,⟨RealStore.columns p c' j.2.1,hj.2⟩)) ho
  exact ⟨congrArg (fun x : Fin p.M × Fin p.N => x.1.val) he,
    congrArg (fun x : Fin p.M × Fin p.N => x.2.val) he⟩

theorem lanes_injective (p : Parameters) (h : Injective p) (r c : Nat) :
    ∀ i j, RealStore.active p r c i → RealStore.active p r c j →
      RealStore.offset p r c i = RealStore.offset p r c j → i=j := by
  rintro ⟨i,j,⟨⟩⟩ ⟨i',j',⟨⟩⟩ hi hj ho
  have he := elements_eq p h r c r c (i,j,PUnit.unit) (i',j',PUnit.unit) hi hj ho
  simp only [RealStore.rows,RealStore.columns,Nat.add_left_cancel_iff] at he
  have hi' : i=i' := Fin.ext he.1
  have hj' : j=j' := Fin.ext he.2
  subst i'
  subst j'
  rfl

theorem rows_recover (p : Parameters) (hBM : 0<p.BM) (r : Nat) (i : Fin p.BM) :
    RealStore.rows p r i / p.BM = r := by
  unfold RealStore.rows
  rw [Nat.add_comm,Nat.add_mul_div_right _ _ hBM,Nat.div_eq_of_lt i.isLt,Nat.zero_add]

theorem columns_recover (p : Parameters) (hBN : 0<p.BN) (c : Nat) (i : Fin p.BN) :
    RealStore.columns p c i / p.BN = c := by
  unfold RealStore.columns
  rw [Nat.add_comm,Nat.add_mul_div_right _ _ hBN,Nat.div_eq_of_lt i.isLt,Nat.zero_add]

/-- Different valid grouped tiles cannot overwrite one another's active
logical output cells under ordinary output-stride injectivity. -/
theorem tiles_disjoint (p : Parameters) (h : Injective p)
    (hBM : 0<p.BM) (hBN : 0<p.BN) (hGM : 0<p.GM)
    (t u : Nat) (ht : t<RealGroup.rows p*RealGroup.columns p)
    (hu : u<RealGroup.rows p*RealGroup.columns p) (hne : t≠u)
    (i j : TileIndex [p.BM,p.BN])
    (hi : RealStore.active p (Schedule.row (RealGroup.rows p) (RealGroup.columns p) p.GM t)
      (Schedule.column (RealGroup.rows p) (RealGroup.columns p) p.GM t) i)
    (hj : RealStore.active p (Schedule.row (RealGroup.rows p) (RealGroup.columns p) p.GM u)
      (Schedule.column (RealGroup.rows p) (RealGroup.columns p) p.GM u) j) :
    RealStore.offset p (Schedule.row (RealGroup.rows p) (RealGroup.columns p) p.GM t)
      (Schedule.column (RealGroup.rows p) (RealGroup.columns p) p.GM t) i ≠
    RealStore.offset p (Schedule.row (RealGroup.rows p) (RealGroup.columns p) p.GM u)
      (Schedule.column (RealGroup.rows p) (RealGroup.columns p) p.GM u) j := by
  intro ho
  have he := elements_eq p h _ _ _ _ i j hi hj ho
  have hr := congrArg (fun n => n/p.BM) he.1
  have hc := congrArg (fun n => n/p.BN) he.2
  simp only [rows_recover p hBM] at hr
  simp only [columns_recover p hBN] at hc
  exact hne (Schedule.grouped_injective _ _ p.GM t u hGM ht hu hr hc)

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealOutputAddresses
