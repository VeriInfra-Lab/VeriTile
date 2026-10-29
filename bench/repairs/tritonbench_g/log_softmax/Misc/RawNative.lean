import bench.repairs.tritonbench_g.log_softmax.Kernels
import bench.repairs.tritonbench_g.log_softmax.Proofs.KernelGrid
import bench.repairs.tritonbench_g.log_softmax.Proofs.Contract
import bench.repairs.tritonbench_g.log_softmax.Proofs.NativeGrid
import VeriTile.Triton.Float.ProfiledMemory

namespace VeriTile.Bench.Repairs.LogSoftmax.RawNative
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open Address
open VeriTile.Bench.Repairs.SoftmaxFlaggems
noncomputable section

/-- Successful complete grids for both actual kernels, all result cells and the
full exterior typed-memory frame. The scalar carrier may be all FP32 words;
the unchanged reducers preserve their selected order and special-value policy. -/
theorem grid_equivalence [Inhabited A] (direction : Direction) (engine : Kernel.Engine A C)
    (M N K BM : Nat) (hb : 0<BM) (hc : 0<C) (covers : N≤C) (initial : State A) :
    ∃ final,
      KernelGrid.run direction engine Bool.false M N K BM (KernelGrid.programs M N K BM) initial=some final ∧
      KernelGrid.run direction engine Bool.true M N K BM (KernelGrid.programs M N K BM) initial=some final ∧
      (∀ q : Coord M N K,final.mem "output" (linear q)=KernelGrid.cell direction engine initial q) ∧
      (∀ (r : RegionName) i,r≠"output" ∨ M*N*K≤ i → final.mem r i=initial.mem r i) := by
  obtain ⟨final,ran,values,frame⟩ := KernelGrid.whole direction engine M N K BM hb hc covers initial
  exact ⟨final,ran,(KernelGrid.candidate_same_run direction engine M N K BM _ initial).symm.trans ran,
    values,frame⟩

abbrev Word := VeriTile.Torch.FP32.Word

/-- The whole canonical FP32 source grids generate identical native-observable
results for every word, under a fixed selected reduction layout, instruction
interpretation and fusion schedule. No finite-value guard or primitive numeric
law is required. Graphs retain all source precision and cast boundaries. -/
theorem forward_native_grid (M N K BM : Nat) (R : Native.Reductions)
    (I : Native.Instructions) (S : Native.Schedule) (memory : RegionName → Nat → Word) :
    ∃ original commuted,
      Observed.run .forward (Native.algebra R) Bool.false M N K BM (Grid.blockColumns N)
        (KernelGrid.programs M N K BM) (NativeGrid.entry memory)=some original ∧
      Observed.run .forward (Native.algebra R) Bool.true M N K BM (Grid.blockColumns N)
        (KernelGrid.programs M N K BM) (NativeGrid.entry memory)=some commuted ∧
      (∀ (r : RegionName) i,Native.observe I S original r i=Native.observe I S commuted r i) ∧
      (∀ (r : RegionName) i,r≠"output" ∨ M*N*K≤ i →
        Native.observe I S original r i=(memory r i).bits ∧
        Native.observe I S commuted r i=(memory r i).bits) := by
  obtain ⟨original,commuted,ra,rb,equal,frame⟩ := NativeGrid.whole .forward R I S
    M N K BM (Grid.blockColumns N) (block_positive N) memory
  refine ⟨original,commuted,ra,rb,equal,?_⟩
  intro r i outside
  apply frame r i
  exact outside

/-- Native observation retains the default enabled FMA contraction. In
particular NativeGrid.backward_fused interprets the actual adjacent sub/mul
site with fmaRN, without equating that schedule to split rounded operations. -/
theorem backward_native_grid (M N K BM : Nat) (R : Native.Reductions)
    (I : Native.Instructions) (S : Native.Schedule) (memory : RegionName → Nat → Word) :
    ∃ original commuted,
      Observed.run .backward (Native.algebra R) Bool.false M N K BM
        (Grid.blockColumns N) (KernelGrid.programs M N K BM)
        (NativeGrid.entry memory)=some original ∧
      Observed.run .backward (Native.algebra R) Bool.true M N K BM
        (Grid.blockColumns N) (KernelGrid.programs M N K BM)
        (NativeGrid.entry memory)=some commuted ∧
      (∀ (r : RegionName) i,Native.observe I S original r i=Native.observe I S commuted r i) ∧
      (∀ (r : RegionName) i,r≠"output" ∨ M*N*K≤ i →
        Native.observe I S original r i=(memory r i).bits ∧
        Native.observe I S commuted r i=(memory r i).bits) := by
  obtain ⟨original,commuted,ra,rb,equal,frame⟩ := NativeGrid.whole .backward R I S
    M N K BM (Grid.blockColumns N) (block_positive N) memory
  refine ⟨original,commuted,ra,rb,equal,?_⟩
  intro r i outside
  apply frame r i
  exact outside


end
end VeriTile.Bench.Repairs.LogSoftmax.RawNative
