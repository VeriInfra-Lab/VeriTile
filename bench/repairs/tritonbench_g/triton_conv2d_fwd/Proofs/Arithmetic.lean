import bench.repairs.tritonbench_g.triton_conv2d_fwd.Spec
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Grid
namespace VeriTile.Bench.Repairs.Conv2d
open scoped BigOperators
variable {α β : Type}

theorem aLoad_reference (s : Shape) (p : ISA α β) (m : Memory α)
    (g r h w c : Nat) (hg : g<s.groups) : aLoad s p m g r h w c=refA s p m g r h w c := by
  have height : (((r/s.ow%s.oh:Nat):Int)*s.sh-s.ph+h)=ih s r h := by unfold ih outH;ring
  have width : (((r%s.ow:Nat):Int)*s.sw-s.pw+w)=iw s r w := by unfold iw outW;ring
  unfold aLoad refA inputArray
  rw [height,width,batch_eq]
  by_cases hc : c<s.cg
  · have channel : g*s.cg+c<s.channels := by unfold Shape.channels;nlinarith
    simp only [hc,channel,true_and,↓reduceIte,Shape.plane]
    split_ifs with active
    · congr 1
      unfold inputPointer
      rw [batch_eq]
      unfold Shape.plane
      have hy := Int.toNat_of_nonneg active.2.1
      have hx := Int.toNat_of_nonneg active.2.2.2.1
      apply Int.ofNat_inj.mp
      rw [Int.toNat_of_nonneg]
      · push_cast
        rw [hy,hx]
        ring
      · have hy0 := active.2.1
        have hx0 := active.2.2.2.1
        have a : (0:Int)≤s.sn := Nat.cast_nonneg _
        have b : (0:Int)≤s.sc := Nat.cast_nonneg _
        have d : (0:Int)≤s.sx := Nat.cast_nonneg _
        have e : (0:Int)≤s.sy := Nat.cast_nonneg _
        positivity
    · rfl
  · simp [hc]

theorem bLoad_reference (s : Shape) (p : ISA α β) (m : Memory α)
    (g f h w c : Nat) : bLoad s p m g f h w c=refB s p m g f h w c := by
  unfold bLoad refB weightArray weightPointer
  have address : s.so*f+s.so*g*s.og+s.tc*c+s.tx*h+s.ty*w=
      s.so*(g*s.og+f)+s.tc*c+s.tx*h+s.ty*w := by ring
  rw [address]

theorem sourceStep_reference (s : Shape) (p : ISA α β) (flags : Flags) (m : Memory α)
    (g r f h w c : Nat) (hg : g<s.groups) (carry : Frag α) :
    sourceStep s p flags m g r f h w c carry=
      referenceStep s p flags m g r f carry (h,w,c) := by
  simp only [sourceStep,referenceStep,aLoad_reference s p m g _ _ _ _ hg,
    bLoad_reference,Nat.mul_comm]

theorem fold_range {γ : Type} (f : γ → Nat → γ) (init : γ) (loop : Nat → γ)
    (zero : loop 0=init) (step : ∀ n,loop (n+1)=f (loop n) n) (n : Nat) :
    loop n=(List.range n).foldl f init := by
  induction n with
  | zero => exact zero
  | succ n ih => rw [step,ih,List.range_succ,List.foldl_append];rfl

theorem channel_reference (s : Shape) (p : ISA α β) (flags : Flags) (m : Memory α)
    (g r f h w c : Nat) (hg : g<s.groups) (carry : Frag α) :
    channelLoop s p flags m g r f h w c carry=
      (List.range c).foldl (fun z c => referenceStep s p flags m g r f z (h,w,c)) carry := by
  apply fold_range (loop := fun n => channelLoop s p flags m g r f h w n carry)
  · rfl
  · intro n
    rw [channelLoop,sourceStep_reference s p flags m g r f h w n hg]

theorem width_reference (s : Shape) (p : ISA α β) (flags : Flags) (m : Memory α)
    (g r f h w : Nat) (hg : g<s.groups) (carry : Frag α) :
    widthLoop s p flags m g r f h w carry=
      ((List.range w).flatMap fun w => (List.range (blocks s.cg 32)).map fun c => (h,w,c)).foldl
        (referenceStep s p flags m g r f) carry := by
  induction w with
  | zero => rfl
  | succ w ih =>
    rw [widthLoop,channel_reference s p flags m g r f h w _ hg,ih,
      List.range_succ,List.flatMap_append,List.foldl_append]
    simp only [List.flatMap_cons,List.flatMap_nil,List.append_nil,List.foldl_map]


theorem source_ordered (s : Shape) (p : ISA α β) (flags : Flags) (m : Memory α)
    (x : Cell) (hg : x.group<s.groups) : sourceValue s p flags m x=ordered s p flags m x := by
  have height : ∀ n,heightLoop s p flags m x.group (x.spatial/64*64) (x.feature/32*32) n=
      ((List.range n).flatMap fun h => (List.range s.kw).flatMap fun w =>
        (List.range (blocks s.cg 32)).map fun c => (h,w,c)).foldl
          (referenceStep s p flags m x.group (x.spatial/64*64) (x.feature/32*32)) (fun _ _ => p.zero) := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih =>
      rw [heightLoop,width_reference s p flags m _ _ _ _ _ hg,ih,
        List.range_succ,List.flatMap_append,List.foldl_append]
      simp only [List.flatMap_cons,List.flatMap_nil,List.append_nil]

  unfold sourceValue ordered taps
  rw [height]

#axiomsClean aLoad_reference
#axiomsClean bLoad_reference
#axiomsClean source_ordered
end VeriTile.Bench.Repairs.Conv2d
