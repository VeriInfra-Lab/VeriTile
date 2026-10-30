import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.RealFiniteContext
import bench.repairs.tritonbench_g.attention_llama.Proofs.AccessSafety
import bench.repairs.tritonbench_g.attention_llama.Proofs.ByteSafety

/-! Finite public-window bounds imply safety of the actual runtime-scalar
source, on every launched CTA and its actual prepared predecessor. Byte
finiteness and mathematical input bindings are unnecessary for access safety.
-/
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealFiniteSafety
open VeriTile Triton Source RealLayout
open VeriTile.Bench.Repairs.AttentionLlama MaskCoverage
open VeriTile.Bench.TritonBenchG.AttentionLlama
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem query_bound (B : Nat) (c : Config) (bounds : RegionBounds)
    (original : BlockState) (vals : (memory B c).Ctx)
    (bounded : (memory B c).InBounds bounds original vals) (i : QueryIndex B c) :
    IOLayout.query c i<bounds c.Q := by
  have access := bounded.1 (0 : Fin 5) (querySlot B c i) (by simp [memory])
  simpa [memory] using access

theorem key_bound (B : Nat) (c : Config) (bounds : RegionBounds)
    (original : BlockState) (vals : (memory B c).Ctx)
    (bounded : (memory B c).InBounds bounds original vals) (i : KeyIndex B c) :
    IOLayout.key c i<bounds c.K := by
  cases selected : c.fp8 with
  | «false» =>
    have access := bounded.1 (1 : Fin 5) (keySlot B c i) (by simp [memory,selected])
    change IOLayout.key c (keyIndex B c (keySlot B c i))<bounds c.K at access
    simpa only [keyIndex_slot] using access
  | «true» =>
    have access := bounded.1 (3 : Fin 5) (keySlot B c i) (by simp [memory,selected])
    change IOLayout.key c (keyIndex B c (keySlot B c i))<bounds c.K at access
    simpa only [keyIndex_slot] using access

theorem value_bound (B : Nat) (c : Config) (bounds : RegionBounds)
    (original : BlockState) (vals : (memory B c).Ctx)
    (bounded : (memory B c).InBounds bounds original vals) (i : KeyIndex B c) :
    IOLayout.value c i<bounds c.V := by
  cases selected : c.fp8 with
  | «false» =>
    have access := bounded.1 (2 : Fin 5) (keySlot B c i) (by simp [memory,selected])
    change IOLayout.value c (keyIndex B c (keySlot B c i))<bounds c.V at access
    simpa only [keyIndex_slot] using access
  | «true» =>
    have access := bounded.1 (4 : Fin 5) (keySlot B c i) (by simp [memory,selected])
    change IOLayout.value c (keyIndex B c (keySlot B c i))<bounds c.V at access
    simpa only [keyIndex_slot] using access

theorem output_bound (B : Nat) (c : Config) (bounds : RegionBounds)
    (original : BlockState) (vals : (memory B c).Ctx)
    (bounded : (memory B c).InBounds bounds original vals) (i : QueryIndex B c) :
    IOLayout.output c i<bounds c.Out := by
  have access := bounded.2.1 (0 : Fin 1) (querySlot B c i) trivial
  simpa [memory] using access

private theorem heads_pos (B : Nat) (c : Config) (s : BlockState)
    (packed : s.pids 1<B*c.NH) : 0<c.NH := by
  by_contra nonpos
  have zero : c.NH=0 := by omega
  simp [zero] at packed

private def batch (B : Nat) (c : Config) (s : BlockState)
    (packed : s.pids 1<B*c.NH) : Fin B :=
  ⟨s.pids 1/c.NH,(Nat.div_lt_iff_lt_mul (heads_pos B c s packed)).mpr packed⟩

private def head (B : Nat) (c : Config) (s : BlockState)
    (packed : s.pids 1<B*c.NH) : Fin c.NH :=
  ⟨s.pids 1%c.NH,Nat.mod_lt _ (heads_pos B c s packed)⟩

theorem query_access (B : Nat) (c : Config) (bounds : RegionBounds)
    (original : BlockState) (vals : (memory B c).Ctx)
    (bounded : (memory B c).InBounds bounds original vals)
    (s : BlockState) (packed : s.pids 1<B*c.NH)
    (i : TileIndex [64,c.D]) (active : alActive s c.H 64 i) :
    alBase s c.NH c.sqz c.sqh+(s.pids 0*64+i.1.val)*c.sqm+i.2.1.val*c.sqk<bounds c.Q := by
  let logical : QueryIndex B c :=
    (batch B c s packed,head B c s packed,⟨s.pids 0*64+i.1.val,active⟩,i.2.1,())
  exact query_bound B c bounds original vals bounded logical

theorem key_access (B : Nat) (c : Config) (bounds : RegionBounds)
    (original : BlockState) (vals : (memory B c).Ctx)
    (bounded : (memory B c).InBounds bounds original vals)
    (s : BlockState) (packed : s.pids 1<B*c.NH)
    (k : Nat) (inside : k<c.N) (d : Fin c.D) :
    alBase s c.NH c.skz c.skh+k*c.skn+d.val*c.skk<bounds c.K := by
  exact key_bound B c bounds original vals bounded
    (batch B c s packed,head B c s packed,⟨k,inside⟩,d,())

theorem value_access (B : Nat) (c : Config) (bounds : RegionBounds)
    (original : BlockState) (vals : (memory B c).Ctx)
    (bounded : (memory B c).InBounds bounds original vals)
    (s : BlockState) (packed : s.pids 1<B*c.NH)
    (k : Nat) (inside : k<c.N) (d : Fin c.D) :
    alBase s c.NH c.svz c.svh+k*c.svk+d.val*c.svn<bounds c.V := by
  exact value_bound B c bounds original vals bounded
    (batch B c s packed,head B c s packed,⟨k,inside⟩,d,())

theorem output_access (B : Nat) (c : Config) (bounds : RegionBounds)
    (original : BlockState) (vals : (memory B c).Ctx)
    (bounded : (memory B c).InBounds bounds original vals)
    (s : BlockState) (packed : s.pids 1<B*c.NH)
    (i : TileIndex [64,c.D]) (active : alActive s c.H 64 i) :
    alOutOffset s c.NH c.soz c.soh c.som c.son 64 i<bounds c.Out := by
  exact output_bound B c bounds original vals bounded
    (batch B c s packed,head B c s packed,⟨s.pids 0*64+i.1.val,active⟩,i.2.1,())

/-- Safety holds for an arbitrary actual source state with the selected
runtime scalar and launch coordinates; no mathematical result is assumed. -/
theorem source_safe (B : Nat) (c : Config) (scale : ℝ) (bounds : RegionBounds)
    (original : BlockState) (vals : (memory B c).Ctx)
    (bounded : (memory B c).InBounds bounds original vals)
    (s : BlockState) (packed : s.pids 1<B*c.NH)
    (bound : RealScalarInvariant.Bound scale s) (undef : ∀ r o,s.undef r o=0)
    (hD : 0<c.D) (hN : 0<c.N) (hSP : c.causal=Bool.true → c.SP=0) :
    Kernel.TraceSafe bounds (Source.kernel c).eraseDType.toAlgKernel s := by
  apply (RealSourceTransport.traceSafe_iff c scale bounds s bound).mpr
  have q := query_access B c bounds original vals bounded s packed
  have k := key_access B c bounds original vals bounded s packed
  have v := value_access B c bounds original vals bounded s packed
  have out := output_access B c bounds original vals bounded s packed
  cases selected : c.fp8 with
  | «false» =>
    have safe := AccessSafety.complete bounds (RealFiniteContext.context c scale s)
      c.Out c.soz c.soh c.som c.son hD hN
      (by intro causal; change c.SP≤s.pids 0*64; rw [hSP causal]; exact Nat.zero_le _)
      undef q k v out
    simpa only [RealSourceTransport.literalSource,selected,Bool.false_eq_true,ite_false,
      RealFiniteContext.context] using safe
  | «true» =>
    have safe := ByteSafety.source_safe c.Q c.K c.V c.Out scale
      c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk
      c.svz c.svh c.svk c.svn c.soz c.soh c.som c.son
      c.NH c.H c.N c.SP 64 64 c.D c.causal bounds s undef q
      (by intro block reached lane d active; exact k _ active d)
      (by intro block reached lane d active; exact v _ active d) out
    simpa only [RealSourceTransport.literalSource,selected,ite_true] using safe

/-- All programs in the actual ordinary grid use the same prepared state.
The source-access conclusion includes both FP16 and byte K/V branches. -/
theorem grid_safe (B : Nat) (c : Config) (scale : ℝ) (bounds : RegionBounds)
    (original : BlockState) (vals : (memory B c).Ctx)
    (bounded : (memory B c).InBounds bounds original vals)
    (hD : 0<c.D) (hN : 0<c.N) (hSP : c.causal=Bool.true → c.SP=0)
    (idx : GridIndex (RealIO.grid B c)) :
    Kernel.TraceSafe bounds (Source.kernel c).eraseDType.toAlgKernel
      (((RealIO.io B c scale).initial vals original).withGridIndex idx) := by
  apply source_safe B c scale bounds original vals bounded
  · have inside := (idx (⟨1,by simp [RealIO.grid,Grid.rank]⟩ : Fin (RealIO.grid B c).rank)).isLt
    simpa [BlockState.withGridIndex,BlockState.withPids,BlockState.withNumPids,
      GridIndex.toPids,RealIO.grid,Grid.rank,Grid.dim] using inside
  · exact RealIO.initial_bound B c scale vals original
  · exact RealIO.initial_undef B c scale vals original
  · exact hD
  · exact hN
  · exact hSP

theorem stage_safe (B : Nat) (c : Config) (scale : ℝ) (bounds : RegionBounds)
    (original : BlockState) (vals : (memory B c).Ctx)
    (bounded : (memory B c).InBounds bounds original vals)
    (hD : 0<c.D) (hN : 0<c.N) (hSP : c.causal=Bool.true → c.SP=0) :
    (AlgorithmStage.ordinary (RealIO.source c) (RealIO.grid B c)).Safe bounds
      ((RealIO.io B c scale).initial vals original) :=
  grid_safe B c scale bounds original vals bounded hD hN hSP

#axiomsClean source_safe
#axiomsClean grid_safe
#axiomsClean stage_safe
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealFiniteSafety
