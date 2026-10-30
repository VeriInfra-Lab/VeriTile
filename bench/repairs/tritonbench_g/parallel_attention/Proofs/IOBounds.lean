import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOInputBinding
import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOSafetyBounds

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure InputBounds (direction : Direction) (shape : Shape) (bounds : RegionName → Nat) : Prop where
  query : ∀ off,off<heads shape*shape.T*shape.K → off<bounds "q"
  key : ∀ off,off<heads shape*shape.T*shape.K → off<bounds "k"
  value : ∀ off,off<heads shape*shape.T*shape.V → off<bounds "v"
  outGrad : direction=.backward → ∀ off,off<heads shape*shape.T*shape.V → off<bounds "do"
  normGrad : direction=.backward → ∀ off,off<heads shape*shape.T → off<bounds "dz"

theorem input_bounds (direction : Direction) (shape : Shape) (jobs : List (Invocation α))
    (xs : ChannelValues α (inputPorts shape)) (bounds : RegionName → Nat)
    (h : (dataIO α direction shape jobs).InBounds bounds xs) : InputBounds direction shape bounds := by
  constructor
  · intro off ho
    exact h.2.1 (⟨0,by decide⟩ : Fin 5) (⟨off,ho⟩,()) (Or.inl (by decide))
  · intro off ho
    exact h.2.1 (⟨1,by decide⟩ : Fin 5) (⟨off,ho⟩,()) (Or.inl (by decide))
  · intro off ho
    exact h.2.1 (⟨2,by decide⟩ : Fin 5) (⟨off,ho⟩,()) (Or.inl (by decide))
  · intro hd off ho
    exact h.2.1 (⟨3,by decide⟩ : Fin 5) (⟨off,ho⟩,()) (Or.inr hd)
  · intro hd off ho
    exact h.2.1 (⟨4,by decide⟩ : Fin 5) (⟨off,ho⟩,()) (Or.inr hd)

private theorem rectangle_dense {bounds : RegionName → Nat} {R : RegionName}
    (M T D head : Nat) (hh : head<M) (h : ∀ off,off<M*T*D → off<bounds R) :
    IOSafety.MatrixBounds bounds R (head*(T*D)) T D D 1 := by
  intro row col hr hc
  apply h
  simpa only [dense,Nat.add_mul,Nat.mul_assoc,Nat.mul_one] using dense_bound T D M head row col hh hr hc

private theorem rectangle_transpose {bounds : RegionName → Nat} {R : RegionName}
    (M T D head : Nat) (hh : head<M) (h : ∀ off,off<M*T*D → off<bounds R) :
    IOSafety.MatrixBounds bounds R (head*(T*D)) D T 1 D := by
  intro row col hr hc
  apply h
  simpa only [dense,Nat.add_mul,Nat.mul_assoc,Nat.mul_one,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    using dense_bound T D M head col row hh hc hr

theorem forward_bounds (shape : Shape) (scale : ℝ) (jobs : List (Invocation α))
    (xs : ChannelValues α (inputPorts shape)) (bounds : RegionName → Nat)
    (h : (dataIO α .forward shape jobs).InBounds bounds xs) (p : Program shape 128) (s : State α) :
    IOSafety.ForwardBounds (forwardConfig shape scale) (withProgram p s) bounds := by
  have hi:=input_bounds .forward shape jobs xs bounds h
  have hv : p.value.val<((shape.V+shape.BV-1)/shape.BV) := p.value.isLt
  constructor
  · exact rectangle_dense (heads shape) shape.T shape.K p.head.val p.head.isLt hi.query
  · exact rectangle_transpose (heads shape) shape.T shape.K p.head.val p.head.isLt hi.key
  · exact rectangle_dense (heads shape) shape.T shape.V p.head.val p.head.isLt hi.value
  · intro i j hr hc
    have hr' : p.chunk.val*128+i.val<shape.T := hr
    have hc' : p.value.val*shape.BV+j.val<shape.V := by
      simpa [forwardConfig,valueChunk,withProgram,Nat.mod_eq_of_lt hv] using hc
    have ho:=h.2.2.1 (⟨0,by decide⟩ : Fin 5) (p.head,p.chunk,p.value,i,j,()) ⟨rfl,hr',hc'⟩
    simpa [dataIO,memory,outputPorts,outputWindow,row,col,dense,heads,forwardConfig,
      queryChunk,valueChunk,withProgram,Nat.div_eq_of_lt hv,Nat.mod_eq_of_lt hv,
      Nat.add_mul,Nat.mul_assoc] using ho
  · intro guard i hr
    have hg : p.value.val=0 := by simpa [forwardConfig,withProgram,Nat.mod_eq_of_lt hv] using guard
    have hr' : p.chunk.val*128+i.val<shape.T := hr
    have ho:=h.2.2.1 (⟨1,by decide⟩ : Fin 5) (p.head,p.chunk,p.value,i,()) ⟨rfl,hg,hr'⟩
    simpa [dataIO,memory,outputPorts,outputWindow,forwardConfig,queryChunk,withProgram,
      Nat.div_eq_of_lt hv,Nat.add_assoc] using ho

theorem backward_bounds (shape : Shape) (scale : ℝ) (jobs : List (Invocation α))
    (xs : ChannelValues α (inputPorts shape)) (bounds : RegionName → Nat)
    (h : (dataIO α .backward shape jobs).InBounds bounds xs) (p : Program shape 64) :
    IOSafety.BackwardBounds (backwardConfig shape scale p.tile) bounds := by
  have hi:=input_bounds .backward shape jobs xs bounds h
  constructor
  · exact rectangle_transpose (heads shape) shape.T shape.K p.head.val p.head.isLt hi.query
  · exact rectangle_dense (heads shape) shape.T shape.K p.head.val p.head.isLt hi.key
  · exact rectangle_dense (heads shape) shape.T shape.V p.head.val p.head.isLt hi.value
  · exact rectangle_transpose (heads shape) shape.T shape.V p.head.val p.head.isLt (hi.outGrad rfl)
  · intro row hr
    apply hi.normGrad rfl
    change p.head.val*shape.T+row<heads shape*shape.T
    have hh:=p.head.isLt
    change p.head.val<heads shape at hh
    change row<shape.T at hr
    nlinarith
  · intro i j hr hc
    have hc' : j.val<shape.K := by simpa [backwardConfig,Program.tile] using hc
    have ho:=h.2.2.1 (⟨3,by decide⟩ : Fin 5) (p.head,p.chunk,p.value,i,j,()) ⟨rfl,hr,hc'⟩
    simpa [dataIO,memory,outputPorts,outputWindow,row,dense,heads,backwardConfig,forwardConfig,Program.tile,
      Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using ho
  · intro i j hr hc
    have ho:=h.2.2.1 (⟨4,by decide⟩ : Fin 5) (p.head,p.chunk,p.value,i,j,()) ⟨rfl,hr,hc⟩
    simpa [dataIO,memory,outputPorts,outputWindow,row,col,dense,heads,backwardConfig,forwardConfig,Program.tile,
      Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using ho

theorem query_bounds (shape : Shape) (scale : ℝ) (jobs : List (Invocation α))
    (xs : ChannelValues α (inputPorts shape)) (bounds : RegionName → Nat)
    (h : (dataIO α .backward shape jobs).InBounds bounds xs) (p : Program shape 64) :
    IOSafety.QueryBounds (dqConfig shape scale p.tile) bounds := by
  have hi:=input_bounds .backward shape jobs xs bounds h
  constructor
  · exact rectangle_dense (heads shape) shape.T shape.K p.head.val p.head.isLt hi.query
  · exact rectangle_dense (heads shape) shape.T shape.K p.head.val p.head.isLt hi.key
  · exact rectangle_transpose (heads shape) shape.T shape.V p.head.val p.head.isLt hi.value
  · exact rectangle_dense (heads shape) shape.T shape.V p.head.val p.head.isLt (hi.outGrad rfl)
  · exact (backward_bounds shape scale jobs xs bounds h p).normGrad
  · intro i j hr hc
    have hc' : j.val<shape.K := by simpa [dqConfig,backwardConfig,Program.tile] using hc
    have ho:=h.2.2.1 (⟨2,by decide⟩ : Fin 5) (p.head,p.chunk,p.value,i,j,()) ⟨rfl,hr,hc'⟩
    simpa [dataIO,memory,outputPorts,outputWindow,row,dense,heads,dqConfig,backwardConfig,forwardConfig,Program.tile,
      Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using ho

#axiomsClean forward_bounds
#axiomsClean backward_bounds
#axiomsClean query_bounds
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
