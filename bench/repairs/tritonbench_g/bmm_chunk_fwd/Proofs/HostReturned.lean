import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.HostInputs

/-! Direct readback of the returned dense tensor in logical coordinates. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host
open VeriTile.Triton
noncomputable section

def returnedOffset (h : Call) (b ch g r n : Nat) : Nat :=
  ((b * h.shape.chunks + ch) * h.shape.groups + g) * h.shape.chunk * h.shape.chunk +
    r * h.shape.chunk + n

/-- Source causal activity is per tile, not an elementwise triangle. -/
def returnedActive (h : Call) (c : Config) (r n : Nat) : Prop :=
  h.causal = Bool.false ∨ n / c.N * c.N < (r / c.M + 1) * c.M

theorem returned_launch (h : Call) (c : Config) (hc : c ∈ configurations)
    (swapped : Bool) (x : Inputs h) (s : BlockState) (hp : Pinned h c x s)
    (A : FlatAlloc) (ha : Allocated h c s A) :
    ∃ final, ∃ _ : Kernel.GridLaunchedOrdinary
      (A.flattenKernel (h.kernel c swapped).eraseDType.toAlgKernel)
      (h.grid c) (A.flattenState s) final,
      (∀ b ch g r n, b < h.shape.batch → ch < h.shape.chunks → g < h.shape.groups →
        r < h.shape.chunk → n < h.shape.chunk → returnedActive h c r n →
        final.readMem A.flat (A.addr (returned h).region (returnedOffset h b ch g r n)) =
          expected h x b ch g r n) ∧
      (∀ r a, ¬ Written h c A r a → final.mem r a = (A.flattenState s).mem r a) := by
  obtain ⟨final, launched, values, frame⟩ := launch h c hc swapped x s hp A ha
  refine ⟨final, launched, ?_, frame⟩
  intro b ch g r n hb hch hg hr hn hactive
  obtain ⟨p, i, hpb, hpg, hpr, hpn⟩ := coverage h c hc b ch g r n hb hch hg hr hn
  have hm := (configuration_positive c hc).1
  have hn' := (configuration_positive c hc).2.1
  have hgroups : 0 < h.shape.groups := Nat.zero_lt_of_lt hg
  have hpc : p2 p / h.shape.groups = ch := by
    rw [hpg, Nat.mul_comm ch, Nat.mul_add_div hgroups]
    simp [Nat.div_eq_of_lt hg]
  have hph : p2 p % h.shape.groups = g := by
    rw [hpg]
    simp [Nat.add_mod, Nat.mod_eq_of_lt hg]
  have hpm : p0 p / RealSetup.tilesN c.N h.parameters = r / c.M := by
    rw [← hpr]
    unfold RealViews.row
    rw [Nat.mul_comm _ c.M, Nat.mul_add_div hm]
    simp [Nat.div_eq_of_lt i.1.isLt]
  have hpn' : p0 p % RealSetup.tilesN c.N h.parameters = n / c.N := by
    rw [← hpn]
    unfold RealViews.column
    rw [Nat.mul_comm _ c.N, Nat.mul_add_div hn']
    simp [Nat.div_eq_of_lt i.2.1.isLt]
  have hi : active h c p i := by
    refine ⟨?_, ?_, ?_⟩
    · change (!h.causal || decide (p0 p % RealSetup.tilesN c.N h.parameters * c.N <
        (p0 p / RealSetup.tilesN c.N h.parameters + 1) * c.M)) = Bool.true
      rw [hpm, hpn']
      simpa [returnedActive] using hactive
    · exact hpr ▸ hr
    · exact hpn ▸ hn
  have hv := values p i hi
  have hoff : offset h c p i = returnedOffset h b ch g r n := by
    unfold offset
    rw [output_offset, hpb, hpg, hpr, hpn]
    unfold returnedOffset
    ring
  rw [hoff] at hv
  simpa only [value, hpb, hpc, hph, hpr, hpn] using hv

end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host

#guard_msgs (drop info) in
#auditModuleAxioms
