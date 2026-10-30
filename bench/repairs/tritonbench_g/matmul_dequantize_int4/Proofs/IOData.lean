import bench.repairs.tritonbench_g.matmul_dequantize_int4.IO

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOData
open VeriTile.Triton FP.Structural
noncomputable section

theorem binding_a {α : Type} [Inhabited α] (c : RawKernel.Layout)
    (jobs : List (Invocation α)) (x : ChannelValues α (inputs c)) (s : State α)
    (h : (program α c jobs).InputBinding x s) (r k : Nat) (hr : r<c.M) (hk : k<c.K) :
    (s.mem "A" (c.A.addr r k)).read .fp16=(arrays c x).A r k := by
  have hv := h (⟨0,by decide⟩ : Fin 4) (⟨r,hr⟩,⟨k,hk⟩,()) trivial
  change (s.mem "A" (c.A.addr r k)).read .fp16=x 0 (⟨r,hr⟩,⟨k,hk⟩,()) at hv
  simpa only [arrays,dif_pos (And.intro hr hk)] using hv

theorem binding_q {α : Type} [Inhabited α] (c : RawKernel.Layout)
    (jobs : List (Invocation α)) (x : ChannelValues α (inputs c)) (s : State α)
    (h : (program α c jobs).InputBinding x s) (k n : Nat) (hk : k<c.K/8) (hn : n<c.N) :
    (s.mem "Q" (c.Q.addr k n)).read .int=(arrays c x).Q k n := by
  have hv := h (⟨1,by decide⟩ : Fin 4) (⟨k,hk⟩,⟨n,hn⟩,()) trivial
  change (s.mem "Q" (c.Q.addr k n)).read .int=x 1 (⟨k,hk⟩,⟨n,hn⟩,()) at hv
  simpa only [arrays,dif_pos (And.intro hk hn)] using hv

theorem binding_s {α : Type} [Inhabited α] (c : RawKernel.Layout)
    (jobs : List (Invocation α)) (x : ChannelValues α (inputs c)) (s : State α)
    (h : (program α c jobs).InputBinding x s) (g n : Nat) (hg : g<c.K/c.group) (hn : n<c.N) :
    (s.mem "S" (c.S.addr g n)).read .fp16=(arrays c x).S g n := by
  have hv := h (⟨2,by decide⟩ : Fin 4) (⟨g,hg⟩,⟨n,hn⟩,()) trivial
  change (s.mem "S" (c.S.addr g n)).read .fp16=x 2 (⟨g,hg⟩,⟨n,hn⟩,()) at hv
  simpa only [arrays,dif_pos (And.intro hg hn)] using hv

theorem binding_z {α : Type} [Inhabited α] (c : RawKernel.Layout)
    (jobs : List (Invocation α)) (x : ChannelValues α (inputs c)) (s : State α)
    (h : (program α c jobs).InputBinding x s) (g n : Nat) (hg : g<c.K/c.group) (hn : n<c.N/8) :
    (s.mem "Z" (c.Z.addr g n)).read .int=(arrays c x).Z g n := by
  have hv := h (⟨3,by decide⟩ : Fin 4) (⟨g,hg⟩,⟨n,hn⟩,()) trivial
  change (s.mem "Z" (c.Z.addr g n)).read .int=x 3 (⟨g,hg⟩,⟨n,hn⟩,()) at hv
  simpa only [arrays,dif_pos (And.intro hg hn)] using hv

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOData
