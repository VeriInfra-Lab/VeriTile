import bench.tritonbench_g.triton_conv2d_fwd.Proofs.KernelBinding
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.Arithmetic

namespace VeriTile.Bench.Repairs.Conv2d.KernelOperands
open VeriTile.Triton FP.Structural KernelBinding
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

def memory (t : State α) : Memory α
  | .input,o => (t.mem "input" o).read .fp32
  | .weight,o => (t.mem "weight" o).read .fp32
  | .output,o => (t.mem "output" o).read .fp32
  | .other _,_ => default

def input (z : α) (p : KernelHalf.Parameters) (t : State α) (h w c : Nat) (i : KernelHalf.AIndex p) : α :=
  if KernelHalf.inputMask p t h w c i then (t.mem p.Input (KernelHalf.inputOffset p t h w c i)).read .fp32 else z

def weight (z : α) (p : KernelHalf.Parameters) (t : State α) (h w c : Nat) (i : KernelHalf.BIndex p) : α :=
  if KernelHalf.weightMask p t c i then (t.mem p.Weight (KernelHalf.weightOffset p t h w c i)).read .fp32 else z

theorem input_load {β} (s : Shape) (v : Valid s) (f : Flags) (pid : Program) (t : State α)
    (p : ISA α β) (h w c : Nat) (i : Fin 128) (k : Fin 32) :
    input p.zero (parameters s f) (atProgram pid t) h w c (i,k,PUnit.unit)=
      aLoad s p (memory t) pid.2.2 (pid.1*128+i.val) h w (c+k.val) := by
  have gd: s.channels/s.groups=s.cg := (group_division s v).1
  simp only [input,KernelHalf.inputMask,KernelHalf.inputOffset,KernelHalf.inputBase,
    KernelHalf.batch,KernelHalf.bh,KernelHalf.bhw,KernelHalf.ih,KernelHalf.iw,
    KernelHalf.oh,KernelHalf.ow,KernelHalf.igd,parameters,KernelHalf.BHW,
    atProgram,gd,aLoad,batch,ih,iw,outH,outW,memory]
  norm_num
  simp only [and_assoc]
  split_ifs with mask
  · have hy : 0≤(h:Int)-s.ph+s.sh*(((pid.1*128+i.val)/s.ow%s.oh : Nat) : Int) := mask.2.2.1
    have hx : 0≤(w:Int)-s.pw+s.sw*(((pid.1*128+i.val)%s.ow : Nat) : Int) := mask.2.2.2.2.1
    congr 2
    unfold inputPointer batch ih iw outH outW
    apply Int.ofNat_inj.mp
    rw [Int.toNat_of_nonneg (by positivity)]
    simp only [Nat.cast_add,Nat.cast_mul]
    push_cast at hy hx ⊢
    rw [Int.toNat_of_nonneg (Int.mul_nonneg (Nat.cast_nonneg _) hy),
      Int.toNat_of_nonneg (Int.mul_nonneg (Nat.cast_nonneg _) hx)]
  · rfl

theorem weight_load {β} (s : Shape) (v : Valid s) (f : Flags) (pid : Program) (t : State α)
    (p : ISA α β) (h w c : Nat) (k j : Fin 32) :
    weight p.zero (parameters s f) (atProgram pid t) h w c (k,j,PUnit.unit)=
      bLoad s p (memory t) pid.2.2 (pid.2.1*32+j.val) h w (c+k.val) := by
  have gd: s.outputs/s.groups=s.og := (group_division s v).2
  simp only [weight,KernelHalf.weightMask,KernelHalf.weightOffset,KernelHalf.weightBase,
    KernelHalf.feat,KernelHalf.ogd,KernelHalf.igd,parameters,KernelHalf.OF,atProgram,gd,(group_division s v).1,bLoad,weightPointer,memory]
  norm_num

#axiomsClean input_load
#axiomsClean weight_load
end
end VeriTile.Bench.Repairs.Conv2d.KernelOperands
