import bench.tritonbench_g.bgmv_shrink_kernel.IO

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOValues
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section
variable {α : Type} [Inhabited α]

/-- Logical arrays are extended only for the existing total mathematical
functions. Active values use exactly their independent tensor coordinates. -/
def input (h : Call) (indices : Nat → Int) (vals : ChannelValues α (IO.inputs h indices))
    (offset : Nat) : α :=
  if hb : offset / h.K < h.batches then if hk : offset % h.K < h.K then
    toFloat (dtype h.storage) (vals 1 (⟨offset/h.K,hb⟩,⟨offset%h.K,hk⟩,PUnit.unit))
  else default else default

def weights (h : Call) (indices : Nat → Int) (vals : ChannelValues α (IO.inputs h indices))
    (offset : Nat) : α :=
  if hs : (offset/h.K)/h.N < IO.slots h indices then
    if hn : (offset/h.K)%h.N < h.N then if hk : offset%h.K < h.K then
      toFloat (dtype h.storage) (vals 2 (⟨(offset/h.K)/h.N,hs⟩,⟨(offset/h.K)%h.N,hn⟩,⟨offset%h.K,hk⟩,PUnit.unit))
    else default else default else default

def output (h : Call) (indices : Nat → Int) (vals : ChannelValues α (IO.inputs h indices))
    (offset : Nat) : α :=
  if hb : offset / h.columns < h.batches then if hn : offset % h.columns < h.N then
    toFloat h.outputType (vals 3 (⟨offset/h.columns,hb⟩,⟨offset%h.columns,hn⟩,PUnit.unit))
  else default else default

def data (h : Call) (indices : Nat → Int) (scale : α)
    (vals : ChannelValues α (IO.inputs h indices)) : Data α where
  input := input h indices vals
  weights := weights h indices vals
  output := output h indices vals
  scaling := scale
  indices := indices
  other := fun _ _ => .mk .real default
  padding := fun d _ _ => defaultValue d

theorem input_value (h : Call) (indices : Nat → Int) (vals : ChannelValues α (IO.inputs h indices))
    (b : Fin h.batches) (k : Fin h.K) :
    input h indices vals (inputOffset h b.val k.val) =
      toFloat (dtype h.storage) (vals 1 (b,k,PUnit.unit)) := by
  have positive : 0 < h.K := Nat.zero_lt_of_lt k.isLt
  simp [input,inputOffset,Nat.add_comm (b.val*h.K),Nat.add_mul_div_right _ _ positive,
    Nat.div_eq_of_lt k.isLt,Nat.mod_eq_of_lt k.isLt,b.isLt,k.isLt]

theorem weight_value (h : Call) (indices : Nat → Int) (vals : ChannelValues α (IO.inputs h indices))
    (slot : Fin (IO.slots h indices)) (n : Fin h.N) (k : Fin h.K) :
    weights h indices vals (weightOffset h slot.val n.val k.val) =
      toFloat (dtype h.storage) (vals 2 (slot,n,k,PUnit.unit)) := by
  have kp : 0 < h.K := Nat.zero_lt_of_lt k.isLt
  have np : 0 < h.N := Nat.zero_lt_of_lt n.isLt
  have address : weightOffset h slot.val n.val k.val = k.val+(n.val+slot.val*h.N)*h.K := by
    unfold weightOffset
    ring
  simp [weights,address,Nat.add_mul_div_right _ _ kp,Nat.add_mul_div_right _ _ np,
    Nat.div_eq_of_lt k.isLt,Nat.div_eq_of_lt n.isLt,Nat.mod_eq_of_lt k.isLt,
    Nat.mod_eq_of_lt n.isLt,slot.isLt,n.isLt,k.isLt]

theorem output_value (h : Call) (indices : Nat → Int) (vals : ChannelValues α (IO.inputs h indices))
    (b : Fin h.batches) (n : Fin h.N) :
    output h indices vals (outputOffset h b.val n.val) =
      toFloat h.outputType (vals 3 (b,n,PUnit.unit)) := by
  have bound : n.val < h.columns := n.isLt.trans_le h.outputCovered
  have positive : 0 < h.columns := Nat.zero_lt_of_lt bound
  simp [output,outputOffset,Nat.add_comm (b.val*h.columns),Nat.add_mul_div_right _ _ positive,
    Nat.div_eq_of_lt bound,Nat.mod_eq_of_lt bound,b.isLt,n.isLt]

#axiomsClean input_value
#axiomsClean weight_value
#axiomsClean output_value
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOValues
