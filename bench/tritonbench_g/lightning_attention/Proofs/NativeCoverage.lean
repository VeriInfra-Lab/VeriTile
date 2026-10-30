import bench.tritonbench_g.lightning_attention.Proofs.NativeTags
import bench.tritonbench_g.lightning_attention.NativeReference

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeCoverage
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.LightningAttention.Native (Word Instructions Plan)
open NativeLocality NativeReference
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (packed_lt)
noncomputable section
set_option maxHeartbeats 6000000

def typed (words : Native.WordMemory) : Memory Word := fun r o => .mk .real (words r o)

theorem samples_binding (ports : List IOChannel)
    (index : (i : Fin ports.length) → TileIndex ports[i].shape → Nat)
    (words : Lane ports → Word) (memory : Memory Word)
    (bound : ∀ lane : Lane ports, memory ports[lane.1].buf (index lane.1 lane.2) = .mk .real (words lane)) :
    Agree (Sites ports index) memory (typed (samples ports index words)) := by
  intro r o hr
  obtain ⟨hreg,hoff⟩ := Classical.choose_spec hr
  change memory r o = .mk .real (samples ports index words r o)
  rw [samples,dif_pos hr]
  exact (congrArg₂ memory hreg hoff).symm.trans (bound (Classical.choose hr))

theorem forward_binding (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (vals : (NativeIO.forward ops plan d).Inputs) (memory : Memory Word)
    (bound : (NativeIO.forward ops plan d).InputCells vals memory) :
    Agree (forwardScope d) memory (typed (forwardWords d vals)) := by
  apply samples_binding
  rintro ⟨i,j⟩
  fin_cases i <;> exact bound _ j True.intro

theorem backward_binding (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (vals : (NativeIO.backward ops plan d).Inputs) (memory : Memory Word)
    (bound : (NativeIO.backward ops plan d).InputCells vals memory) :
    Agree (backwardScope d) memory (typed (backwardWords d vals)) := by
  apply samples_binding
  rintro ⟨i,j⟩
  fin_cases i <;> exact bound _ j True.intro

theorem forwardCovered (d : Host.Dimensions) (pid : PID) (hp : ValidPID .forward d pid) :
    ForwardCovered (forwardScope d) d.parameters pid := by
  refine ⟨?_,?_,?_⟩
  · intro t a ht ha
    exact ⟨⟨⟨0,by change 0<4; decide⟩,(⟨pid.head,hp.1⟩,⟨t,ht⟩,⟨a,ha⟩,())⟩,rfl,rfl⟩
  · intro t a ht ha
    exact ⟨⟨⟨1,by change 1<4; decide⟩,(⟨pid.head,hp.1⟩,⟨t,ht⟩,⟨a,ha⟩,())⟩,rfl,rfl⟩
  · intro t c ht hc
    exact ⟨⟨⟨2,by change 2<4; decide⟩,(⟨pid.head,hp.1⟩,⟨t,ht⟩,⟨pid.channel*d.BM+c,packed_lt hp.2 hc⟩,())⟩,rfl,rfl⟩

theorem backwardCovered (d : Host.Dimensions) (head : Nat) (hh : head<d.B*d.H) :
    HeadCovered (backwardScope d) d.parameters head := by
  refine ⟨?_,?_,?_,?_,?_,?_,?_⟩
  · intro t a ht ha
    exact ⟨⟨⟨0,by change 0<7; decide⟩,(⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,())⟩,rfl,rfl⟩
  · intro t a ht ha
    exact ⟨⟨⟨1,by change 1<7; decide⟩,(⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,())⟩,rfl,rfl⟩
  · intro t a ht ha
    exact ⟨⟨⟨2,by change 2<7; decide⟩,(⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,())⟩,rfl,rfl⟩
  · intro t a ht ha
    exact ⟨⟨⟨3,by change 3<7; decide⟩,(⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,())⟩,rfl,rfl⟩
  · intro t a ht ha
    exact ⟨⟨⟨4,by change 4<7; decide⟩,(⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,())⟩,rfl,rfl⟩
  · intro t a ht ha
    exact ⟨⟨⟨5,by change 5<7; decide⟩,(⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,())⟩,rfl,rfl⟩
  · intro t a ht ha
    exact ⟨⟨⟨6,by change 6<7; decide⟩,(⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,())⟩,rfl,rfl⟩

theorem forward_output_scope (d : Host.Dimensions)
    (j : TileIndex [d.B*d.H,d.N,NativeIO.paddedWidth d]) :
    forwardScope d "output" (NativeIO.paddedIndex d j) :=
  ⟨⟨⟨3,by change 3<4; decide⟩,j⟩,rfl,rfl⟩

theorem backward_output_scope (d : Host.Dimensions) (o : Fin (backwardChannels d).length)
    (j : TileIndex (backwardChannels d)[o].shape) :
    backwardScope d (backwardChannels d)[o].buf (backwardIndex d o j) := by
  fin_cases o
  · exact ⟨⟨⟨4,by change 4<7; decide⟩,j⟩,rfl,rfl⟩
  · exact ⟨⟨⟨5,by change 5<7; decide⟩,j⟩,rfl,rfl⟩
  · exact ⟨⟨⟨6,by change 6<7; decide⟩,j⟩,rfl,rfl⟩

theorem forward_outside (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (vals : (NativeIO.forward ops plan d).Inputs) (r : RegionName) (o : Nat)
    (outside : (NativeIO.forward ops plan d).Outside vals r o)
    (pid : PID) (hp : ValidPID .forward d pid) :
    ¬Footprint .forward d pid r o := by
  rintro ⟨rfl,t,ht,c,hc,ho⟩
  let j : TileIndex [d.B*d.H,d.N,NativeIO.paddedWidth d] :=
    (⟨pid.head,hp.1⟩,⟨t,ht⟩,⟨pid.channel*d.BM+c,packed_lt hp.2 hc⟩,())
  have no := outside.1 ⟨0,by change 0<1; decide⟩ j True.intro
  rcases no with hr|hoff
  · exact hr rfl
  · apply hoff
    simpa only [NativeIO.forward,NativeIO.paddedIndex,j,Geometry.dense,Nat.add_mul] using ho

theorem backward_outside (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (vals : (NativeIO.backward ops plan d).Inputs) (r : RegionName) (o : Nat)
    (outside : (NativeIO.backward ops plan d).Outside vals r o)
    (stage : Stage) (hs : stage=.intra ∨ stage=.inter)
    (pid : PID) (hp : ValidPID stage d pid) :
    ¬Footprint stage d pid r o := by
  have hh : pid.head<d.B*d.H := by
    have hh := hp.1
    cases stage <;> exact hh
  have noQ : ¬HeadFoot d.N d.D pid.head "queryGradient" r o := by
    rintro ⟨rfl,t,ht,a,ha,ho⟩
    have no := outside.1 ⟨0,by change 0<3; decide⟩ (⟨pid.head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) True.intro
    rcases no with hr|hoff
    · exact hr rfl
    · apply hoff
      simpa only [NativeIO.backward,backwardIndex,denseIndex,Geometry.dense,Nat.add_mul] using ho
  have noK : ¬HeadFoot d.N d.D pid.head "keyGradient" r o := by
    rintro ⟨rfl,t,ht,a,ha,ho⟩
    have no := outside.1 ⟨1,by change 1<3; decide⟩ (⟨pid.head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) True.intro
    rcases no with hr|hoff
    · exact hr rfl
    · apply hoff
      simpa only [NativeIO.backward,backwardIndex,denseIndex,Geometry.dense,Nat.add_mul] using ho
  have noV : ¬HeadFoot d.N d.E pid.head "valueGradient" r o := by
    rintro ⟨rfl,t,ht,a,ha,ho⟩
    have no := outside.1 ⟨2,by change 2<3; decide⟩ (⟨pid.head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) True.intro
    rcases no with hr|hoff
    · exact hr rfl
    · apply hoff
      simpa only [NativeIO.backward,backwardIndex,denseIndex,Geometry.dense,Nat.add_mul] using ho
  rcases hs with rfl|rfl
  · rintro (hq|hk|hv)
    · exact noQ (rows_head _ _ _ _ _ _ _ _ hq)
    · exact noK (rows_head _ _ _ _ _ _ _ _ hk)
    · exact noV (rows_head _ _ _ _ _ _ _ _ hv)
  · rintro (hq|hk|hv)
    · exact noQ hq
    · exact noK hk
    · exact noV hv

end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeCoverage

