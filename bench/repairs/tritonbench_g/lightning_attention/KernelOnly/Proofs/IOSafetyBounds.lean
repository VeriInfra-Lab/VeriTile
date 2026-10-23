import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.IO
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Geometry
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.SourceSafetyBounds
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.IOProof
open VeriTile Triton FP.Structural
noncomputable section

theorem forward_bounds (d : Host.Dimensions) (geometry : Host.Contract d)
    (x : (forwardIO d).Inputs) (bounds : RegionName → Nat) (h : (forwardIO d).InBounds bounds x)
    (pid : PID) (valid : ValidPID .forward d pid) :
    SourceSafety.ForwardBounds d.parameters pid bounds := by
  refine ⟨?_,?_,?_,?_⟩
  · intro t a ht ha
    exact h.2.1 (0 : Fin 4) (⟨pid.head,valid.1⟩,⟨t,ht⟩,⟨a,ha⟩,()) (by change 0 < 3; omega)
  · intro t a ht ha
    exact h.2.1 (1 : Fin 4) (⟨pid.head,valid.1⟩,⟨t,ht⟩,⟨a,ha⟩,()) (by change 1 < 3; omega)
  · intro t c ht hc
    have fc := Geometry.channel_active d geometry pid.channel valid.2 ⟨c,hc⟩
    have haddr := h.2.1 (2 : Fin 4) (⟨pid.head,valid.1⟩,⟨t,ht⟩,⟨pid.channel*d.BM+c,fc⟩,()) (by change 2 < 3; omega)
    change pid.head*d.N*d.E+t*d.E+(pid.channel*d.BM+c) < bounds "value" at haddr
    change pid.head*d.N*d.E+pid.channel*d.BM+t*d.E+c < bounds "value"
    omega
  · intro t c ht hc
    have fc := Geometry.channel_active d geometry pid.channel valid.2 ⟨c,hc⟩
    have haddr := h.2.2.1 (0 : Fin 1) (⟨pid.head,valid.1⟩,⟨t,ht⟩,⟨pid.channel*d.BM+c,fc⟩,()) trivial
    change pid.head*d.N*d.E+t*d.E+(pid.channel*d.BM+c) < bounds "output" at haddr
    change pid.head*d.N*d.E+pid.channel*d.BM+t*d.E+c < bounds "output"
    omega

theorem backward_bounds (d : Host.Dimensions) (x : (backwardIO d).Inputs)
    (bounds : RegionName → Nat) (h : (backwardIO d).InBounds bounds x)
    (head : Nat) (hh : head < d.B*d.H) : SourceSafety.HeadBounds d.parameters head bounds := by
  refine ⟨?_,?_,?_,?_,?_,?_,?_⟩
  · intro t a ht ha; exact h.2.1 (0 : Fin 4) (⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) trivial
  · intro t a ht ha; exact h.2.1 (1 : Fin 4) (⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) trivial
  · intro t c ht hc; exact h.2.1 (2 : Fin 4) (⟨head,hh⟩,⟨t,ht⟩,⟨c,hc⟩,()) trivial
  · intro t c ht hc; exact h.2.1 (3 : Fin 4) (⟨head,hh⟩,⟨t,ht⟩,⟨c,hc⟩,()) trivial
  · intro t a ht ha; exact h.2.2.1 (0 : Fin 3) (⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) trivial
  · intro t a ht ha; exact h.2.2.1 (1 : Fin 3) (⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) trivial
  · intro t c ht hc; exact h.2.2.1 (2 : Fin 3) (⟨head,hh⟩,⟨t,ht⟩,⟨c,hc⟩,()) trivial
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.IOProof
