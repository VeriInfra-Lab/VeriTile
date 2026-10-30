import bench.tritonbench_g.attention_llama.Proofs.RealBacking

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealGridResult
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionLlama LoopInvariant HostGrid HostResult Correctness
open VeriTile.Bench.TritonBenchG.AttentionLlama
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open Backing
noncomputable section

/-- Complete FP16/E5M2 physical grid over raw regional state and numeric
strides. Every active lane has the independent mathematical attention value;
every other typed cell is unchanged by its source program. -/
theorem attention_kernel_grid (B : Nat) (c : Context) (Out : RegionName)
    (layout : OutputLayout B c) (fp8 : Bool)
    (hD : 0<c.D) (hN : 0<c.N) (hSP : c.causal=Bool.true → c.SP=0)
    (hu : ∀ r offset,c.initial.undef r offset=0)
    (domain : fp8=Bool.true → ByteHostResult.Domain (B:=B) (heads:=c.NH) c)
    (sep : fp8=Bool.true → c.Q≠c.K ∧ c.Q≠c.V ∧ Out≠c.K ∧ Out≠c.V) :
    ∃ states : Program B c.NH c.H→BlockState,
      (∀ program, exec (realSource c Out layout.soz layout.soh layout.som layout.son fp8).eraseDType.toAlgKernel
        (contextFor c (anchor hD program)).initial=some (states program)) ∧
      (∀ program (j : TileIndex [64,c.D]),
        alActive (contextFor c (anchor hD program)).initial c.H 64 j →
        (states program).readMem Out
          (alOutOffset (contextFor c (anchor hD program)).initial c.NH
            layout.soz layout.soh layout.som layout.son 64 j)=
          RealKernel.target (realView (contextFor c (anchor hD program)) fp8) j) ∧
      (∀ program region offset, (region≠Out ∨ ∀ j : TileIndex [64,c.D],
        alActive (contextFor c (anchor hD program)).initial c.H 64 j →
          offset≠alOutOffset (contextFor c (anchor hD program)).initial c.NH
            layout.soz layout.soh layout.som layout.son 64 j) →
        (states program).mem region offset=c.initial.mem region offset) := by
  have runs := fun program : Program B c.NH c.H =>
    real_block c Out layout.soz layout.soh layout.som layout.son fp8 hD hN hSP hu layout
      ⟨rfl,rfl,rfl,rfl⟩ domain sep (anchor hD program)
  choose states he hv hf using runs
  exact ⟨states,he,hv,hf⟩

#axiomsClean attention_kernel_grid
#specNonCircular RealKernel.target avoiding [realSource,Triton.exec]
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealGridResult
