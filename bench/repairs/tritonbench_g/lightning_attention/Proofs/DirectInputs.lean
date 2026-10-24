import bench.repairs.tritonbench_g.lightning_attention.Proofs.HostIngress
namespace VeriTile.Bench.Repairs.LightningAttention.Checked
open VeriTile Triton FP.Structural Host
noncomputable section
set_option maxHeartbeats 2000000

/-- Scalar address consequence of the ordinary contiguous-layout rule.
Singleton axes may retain arbitrary strides, since their only coordinate is zero. -/
theorem offset4_contiguous (d : Dimensions) (src : Tensor) (width b h t f : Nat)
    (hs : src.shape=[d.B,d.H,d.N,width]) (layout : DenseLayout src.shape src.strides)
    (hb : b<d.B) (hh : h<d.H) (ht : t<d.N) (hf : f<width) :
    offset4 src b h t f=packed d width b h t f := by
  rw [hs] at layout
  cases hs0 : src.strides with
  | nil => simp [hs0,DenseLayout] at layout
  | cons s0 tail0 =>
    cases tail0 with
    | nil => simp [hs0,DenseLayout] at layout
    | cons s1 tail1 =>
      cases tail1 with
      | nil => simp [hs0,DenseLayout] at layout
      | cons s2 tail2 =>
        cases tail2 with
        | nil => simp [hs0,DenseLayout] at layout
        | cons s3 tail3 =>
          cases tail3 with
          | cons x xs => simp [hs0,DenseLayout] at layout
          | nil =>
            simp only [hs0,DenseLayout,List.prod_cons,List.prod_nil,Nat.mul_one] at layout
            rcases layout with ⟨h0,h1,h2,h3,_⟩
            have e0 : b*s0=b*(d.H*d.N*width) := by
              rcases h0 with h0|h0
              · have : b=0 := by omega
                simp [this]
              · rw [h0];ring
            have e1 : h*s1=h*(d.N*width) := by
              rcases h1 with h1|h1
              · have : h=0 := by omega
                simp [this]
              · rw [h1]
            have e2 : t*s2=t*width := by
              rcases h2 with h2|h2
              · have : t=0 := by omega
                simp [this]
              · rw [h2]
            have e3 : f*s3=f := by
              rcases h3 with h3|h3
              · have : f=0 := by omega
                simp [this]
              · simp [h3]
            simp only [offset4,hs0,List.getElem!_cons_zero,List.getElem!_cons_succ]
            rw [e0,e1,e2,e3]
            unfold packed
            ring

/-- The three saved descriptors are exactly the caller's input Tensor objects.
No forward materialization or separate input memory appears. -/
structure Direct (c : Invocation) (a : Allocations) : Prop where
  query : a.queryDense=c.query.tensor
  key : a.keyDense=c.key.tensor
  value : a.valueDense=c.value.tensor
  resolved : c.query.lazyNeg=Bool.false ∧ c.key.lazyNeg=Bool.false ∧ c.value.lazyNeg=Bool.false
  queryShape : c.query.tensor.shape=Host.queryShape c.dims
  keyShape : c.key.tensor.shape=Host.queryShape c.dims
  valueShape : c.value.tensor.shape=Host.valueShape c.dims
  layouts : DenseLayout c.query.tensor.shape c.query.tensor.strides ∧
    DenseLayout c.key.tensor.shape c.key.tensor.strides ∧
    DenseLayout c.value.tensor.shape c.value.tensor.strides

theorem direct_reads {α : Type} [Inhabited α] (negate : α→α) (c : Invocation)
    (a : Allocations) (direct : Direct c a) (memory : Nat → Cell α) :
    ForwardCopies negate c a memory memory := by
  constructor
  · intro b h t f
    have he:=offset4_contiguous c.dims c.query.tensor c.dims.D b h t f direct.queryShape direct.layouts.1 b.isLt h.isLt t.isLt f.isLt
    simp [callerRead,direct.query,direct.resolved.1,he]
  · intro b h t f
    have he:=offset4_contiguous c.dims c.key.tensor c.dims.D b h t f direct.keyShape direct.layouts.2.1 b.isLt h.isLt t.isLt f.isLt
    simp [callerRead,direct.key,direct.resolved.2.1,he]
  · intro b h t f
    have he:=offset4_contiguous c.dims c.value.tensor c.dims.E b h t f direct.valueShape direct.layouts.2.2 b.isLt h.isLt t.isLt f.isLt
    simp [callerRead,direct.value,direct.resolved.2.2,he]

#axiomsClean offset4_contiguous
#axiomsClean direct_reads
#specNonCircular Direct avoiding [fwdMemory,fwdOutput,fwdCarry]
end
end VeriTile.Bench.Repairs.LightningAttention.Checked
