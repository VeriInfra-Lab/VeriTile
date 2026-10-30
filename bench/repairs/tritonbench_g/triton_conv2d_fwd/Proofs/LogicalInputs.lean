import bench.repairs.tritonbench_g.triton_conv2d_fwd.IO

namespace VeriTile.Bench.Repairs.Conv2d.ConvIO
open VeriTile Triton FP.Structural KernelNumeric KernelBinding
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

def Occurs (s : Shape) (r : RegionName) (off : Nat) : Prop :=
  ∃ pos : InputPosition s, (inputChannels s)[pos.1].buf=r ∧ readAddress s pos.1 pos.2=off

theorem read_extension (o : Operations A) (s : Shape) (f : Flags) (vals : Inputs A s)
    (t : State (Payload A)) (bound : (program o s f).InputBinding vals t)
    (r : RegionName) (off : Nat) (occurs : Occurs s r off) :
    (t.mem r off).read .fp32=logicalRead (.scalar o.zero) s vals r off := by
  have all : ∀ i j, (t.mem (inputChannels s)[i].buf (readAddress s i j)).read .fp32=payload s vals i j := by
    intro i j
    fin_cases i <;> simpa [program,memory,inputChannels,payload,mapValue] using bound _ j trivial
  unfold Occurs at occurs
  unfold logicalRead
  rw [dif_pos occurs]
  obtain ⟨hr,ho⟩ := Classical.choose_spec occurs
  simpa only [hr,ho] using all (Classical.choose occurs).1 (Classical.choose occurs).2

theorem occurs_input (s : Shape) (n c y x : Nat)
    (hn : n < s.n) (hc : c < s.channels) (hy : y < s.h) (hx : x < s.w) :
    Occurs s "input" (s.sn*n+s.sc*c+s.sx*y+s.sy*x) :=
  ⟨⟨⟨0,by simp [inputChannels]⟩,(⟨n,hn⟩,⟨c,hc⟩,⟨y,hy⟩,⟨x,hx⟩,())⟩,rfl,rfl⟩

theorem occurs_weight (s : Shape) (o c h w : Nat)
    (ho : o < s.outputs) (hc : c < s.cg) (hh : h < s.kh) (hw : w < s.kw) :
    Occurs s "weight" (s.so*o+s.tc*c+s.tx*h+s.ty*w) :=
  ⟨⟨⟨1,by simp [inputChannels]⟩,(⟨o,ho⟩,⟨c,hc⟩,⟨h,hh⟩,⟨w,hw⟩,())⟩,rfl,rfl⟩

/-- The source's signed spatial expressions are converted only at active
nonnegative coordinates. Their actual pointer is a declared tensor window. -/
theorem source_input_occurs (s : Shape) (v : Valid s) (f : Flags) (pid : Program)
    (hg : pid.2.2 < s.groups) (t : State A) (h w c : Nat) (i : TileIndex [128,32])
    (active : KernelHalf.inputMask (parameters s f) (atProgram pid t) h w c i) :
    Occurs s "input" (KernelHalf.inputOffset (parameters s f) (atProgram pid t) h w c i) := by
  let r := pid.1*128+i.1.val
  have mask : batch s r < s.n ∧ c+i.2.1.val < s.cg ∧
      0 ≤ ih s r h ∧ ih s r h < s.h ∧ 0 ≤ iw s r w ∧ iw s r w < s.w := by
    simpa [KernelHalf.inputMask,KernelHalf.batch,KernelHalf.bh,KernelHalf.bhw,
      KernelHalf.ih,KernelHalf.iw,KernelHalf.oh,KernelHalf.ow,KernelHalf.igd,
      parameters,atProgram,KernelHalf.BHW,batch,ih,iw,outH,outW,r,
      (group_division s v).1,and_assoc] using active
  have channel : pid.2.2*s.cg+(c+i.2.1.val) < s.channels := by
    unfold Shape.channels
    nlinarith [mask.2.1]
  have hy : (ih s r h).toNat < s.h := by omega
  have hx : (iw s r w).toNat < s.w := by omega
  have occ := occurs_input s (batch s r) (pid.2.2*s.cg+(c+i.2.1.val))
    (ih s r h).toNat (iw s r w).toNat mask.1 channel hy hx
  convert occ using 1
  congr 1
  simp only [KernelHalf.inputOffset,KernelHalf.inputBase,KernelHalf.batch,KernelHalf.bh,
    KernelHalf.bhw,KernelHalf.ih,KernelHalf.iw,KernelHalf.oh,KernelHalf.ow,KernelHalf.igd,
    parameters,atProgram,KernelHalf.BHW,reduceIte,(group_division s v).1,
    batch,ih,iw,outH,outW,r]
  norm_num
  change s.sn*batch s r+s.sc*pid.2.2*s.cg+s.sc*(c+i.2.1.val)+
    ((s.sx:Int)*ih s r h).toNat+((s.sy:Int)*iw s r w).toNat =
    s.sn*batch s r+s.sc*(pid.2.2*s.cg+(c+i.2.1.val))+s.sx*(ih s r h).toNat+s.sy*(iw s r w).toNat
  rw [Int.toNat_mul (Nat.cast_nonneg _) mask.2.2.1,
    Int.toNat_mul (Nat.cast_nonneg _) mask.2.2.2.2.1]
  simp only [Int.toNat_natCast]
  ring

theorem source_weight_occurs (s : Shape) (v : Valid s) (f : Flags) (pid : Program)
    (hg : pid.2.2 < s.groups) (t : State A) (h w c : Nat)
    (hh : h < s.kh) (hw : w < s.kw) (i : TileIndex [32,32])
    (active : KernelHalf.weightMask (parameters s f) (atProgram pid t) c i) :
    Occurs s "weight" (KernelHalf.weightOffset (parameters s f) (atProgram pid t) h w c i) := by
  have mask : c+i.1.val < s.cg ∧ pid.2.1*32+i.2.1.val < s.og := by
    simpa [KernelHalf.weightMask,KernelHalf.igd,KernelHalf.ogd,KernelHalf.feat,
      parameters,atProgram,KernelHalf.OF,(group_division s v).1,(group_division s v).2] using active
  have feature : pid.2.2*s.og+(pid.2.1*32+i.2.1.val) < s.outputs := by
    unfold Shape.outputs
    nlinarith [mask.2]
  have occ := occurs_weight s (pid.2.2*s.og+(pid.2.1*32+i.2.1.val))
    (c+i.1.val) h w feature mask.1 hh hw
  convert occ using 1
  congr 1
  simp only [KernelHalf.weightOffset,KernelHalf.weightBase,KernelHalf.feat,KernelHalf.ogd,
    parameters,atProgram,KernelHalf.OF,reduceIte,(group_division s v).2]
  norm_num
  ring

theorem active_output (s : Shape) (j : OutputLane s) : s.Active (outputCell s j) :=
  ⟨j.1.isLt,j.2.1.isLt,j.2.2.1.isLt⟩

theorem output_covers (s : Shape) (v : Valid s) (a : Nat) (ha : a < s.size) :
    ∃ j : OutputLane s, outputAddress s j=a := by
  obtain ⟨x,hx,he⟩ := address_covers s v a ha
  exact ⟨(⟨x.group,hx.1⟩,⟨x.spatial,hx.2.1⟩,⟨x.feature,hx.2.2⟩,()),he⟩

end
end VeriTile.Bench.Repairs.Conv2d.ConvIO
