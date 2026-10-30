import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.IO
import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.ClosedSafety

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO
open Whole001 Source VeriTile.Triton Concurrent
noncomputable section
set_option maxHeartbeats 5000000

theorem arrayWindow_bound (name : String) (region size : Nat) (sizes : Nat → Nat)
    (h : (arrayWindow name region size).InBounds sizes) : size ≤ sizes region := by
  by_cases hz : size=0
  · omega
  · have positive : 0<size := by omega
    let last : Fin size := ⟨size-1,by omega⟩
    have bound := h last rfl
    change 0 ≤ (0 : Int)+(last.val:Int)*1 ∧ (0 : Int)+(last.val:Int)*1 < sizes region at bound
    simp only [Int.zero_add,Int.mul_one] at bound
    have less : last.val < sizes region := by exact_mod_cast bound.2
    dsimp only [last] at less
    omega

theorem input_bounds (p : Parameters) (observe : V → O) (bs : Concurrent.Bounds)
    (h : (io p observe).InBounds bs) :
    (∀ r,(ClosedSource.bounds p).input r ≤ bs.input r) ∧
    (∀ r,(ClosedSource.bounds p).mem r ≤ bs.mem r) ∧
    (∀ r,(ClosedSource.bounds p).words r ≤ bs.words r) := by
  have ha := h.1 ⟨.input,aWindow p⟩ (by simp [io,scalarInputs])
  have hb := h.1 ⟨.input,bWindow p⟩ (by simp [io,scalarInputs])
  have hc := h.1 ⟨.mem,cWindow p⟩ (by simp [io,scalarInputs])
  have hs := h.1 ⟨.mem,sInitialWindow p⟩ (by simp [io,scalarInputs])
  have a := arrayWindow_bound _ _ _ bs.input ha
  have b := arrayWindow_bound _ _ _ bs.input hb
  have c := arrayWindow_bound _ _ _ bs.mem hc
  have s := arrayWindow_bound _ _ _ bs.mem hs
  refine ⟨?_,?_,?_⟩
  · intro r
    by_cases h0 : r=0
    · subst r; exact a
    · by_cases h1 : r=1
      · subst r; exact b
      · simp [ClosedSource.bounds,h0,h1]
  · intro r
    by_cases h0 : r=0
    · subst r; exact c
    · by_cases h1 : r=1
      · subst r; exact s
      · simp [ClosedSource.bounds,h0,h1]
  · intro r; exact Nat.zero_le _

theorem c_written (p : Parameters) (observe : V → O) (i : C p) :
    (io p observe).ScalarWritten 0 i.val := by
  refine ⟨cWindow p,by simp [io],i,rfl,rfl,?_⟩
  simp [Window.address,cWindow,arrayWindow]

theorem s_written (p : Parameters) (observe : V → O) (i : S p)
    (active : 0<(Plan.select p.h).streamTiles) : (io p observe).ScalarWritten 1 i.val := by
  refine ⟨sWindow p,by simp [io],i,?_,rfl,?_⟩
  · simpa only [sWindow] using (decide_eq_true active)
  · simp [Window.address,sWindow,sInitialWindow,arrayWindow]

#axiomsClean input_bounds
#axiomsClean c_written
#axiomsClean s_written
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO
