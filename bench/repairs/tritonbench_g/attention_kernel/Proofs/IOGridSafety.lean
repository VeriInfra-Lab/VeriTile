import bench.repairs.tritonbench_g.attention_kernel.IO
import bench.repairs.tritonbench_g.attention_kernel.Proofs.IOSourceSafety

namespace VeriTile.Bench.Repairs.AttentionKernel.IOGridSafety
open VeriTile Triton FP.Structural
open Whole001 Source Whole Machine
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
variable {α β : Type} [Inhabited β]

theorem preparation (M : Algebra β) (dot : Dot β) (p : Parameters) (a : Program) (s : State β) :
    (IO.job M dot p a).initial s=start (grid p) a s := rfl

theorem schedule (M : Algebra β) (dot : Dot β) (p : Parameters) (work : List Program) (s : State β) :
    runSchedule (work.map (IO.job M dot p)) s=
      runPrograms (fun _=>M) (fun _=>dot) p (grid p) work s := by
  induction work generalizing s with
  | nil=>rfl
  | cons a rest ih=>
    simp only [List.map_cons,runSchedule,runPrograms]
    change (FP.Structural.exec _ _ (start (grid p) a s)).bind _ =
      (FP.Structural.exec _ _ (start (grid p) a s)).bind _
    congr 1
    funext t
    exact ih t

theorem bounds (M : Algebra β) (dot : Dot β) (p : Parameters) (work : List Program)
    (encode : α → β) (observe : TileIndex (IO.TensorShape p) → β → α)
    (vals : ChannelValues α (IO.inputs p)) (bounds : RegionName → Nat)
    (bounded : (IO.attentionIO M dot p work encode observe).InBounds bounds vals)
    (a : Program) (member : a∈programs (grid p)) (s : State β) :
    IOSafety.Bounds p (start (grid p) a s) bounds := by
  obtain ⟨_,hy,_⟩:=(mem_programs (grid p) a.1 a.2.1 a.2.2).mp member
  obtain ⟨_,hb,hh⟩:=head_decode p a.2.1 hy
  let b : Fin p.batches:=⟨a.2.1/p.heads,hb⟩
  let h : Fin p.heads:=⟨a.2.1%p.heads,hh⟩
  constructor
  · intro row hr d hd
    exact bounded.2.1 (0 : Fin 4) (b,h,⟨row,hr⟩,⟨d,hd⟩,()) trivial
  · intro row hr d hd
    exact bounded.2.1 (1 : Fin 4) (b,h,⟨row,hr⟩,⟨d,hd⟩,()) trivial
  · intro row hr d hd
    exact bounded.2.1 (2 : Fin 4) (b,h,⟨row,hr⟩,⟨d,hd⟩,()) trivial
  · intro row hr column hc
    exact bounded.2.1 (3 : Fin 4) (b,h,⟨row,hr⟩,⟨column,hc⟩,()) trivial
  · intro row hr d hd
    exact bounded.2.2.1 (0 : Fin 1) (b,h,⟨row,hr⟩,⟨d,hd⟩,()) trivial

theorem safe (M : Algebra β) (dot : Dot β) (p : Parameters)
    (bases : Bases) (domain : Domain p bases) (hbm : 0<p.bm) (work : List Program)
    (members : ∀ a∈work,a∈programs (grid p))
    (encode : α → β) (observe : TileIndex (IO.TensorShape p) → β → α)
    (vals : ChannelValues α (IO.inputs p)) (bounds : RegionName → Nat)
    (bounded : (IO.attentionIO M dot p work encode observe).InBounds bounds vals) (s : State β) :
    ScheduleSafe bounds (work.map (IO.job M dot p)) s := by
  have bound (a : Program) (ha : a∈programs (grid p)) (s : State β):=
    IOGridSafety.bounds M dot p work encode observe vals bounds bounded a ha s
  clear bounded vals
  induction work generalizing s with
  | nil=>trivial
  | cons a rest ih=>
    have member:=members a (by simp)
    have live : Live p := by
      by_contra empty
      rw [RawKernel.no_grid p hbm empty] at member
      exact List.not_mem_nil member
    refine ⟨?_,?_⟩
    · change TraceSafe _ (bounds) (kernel p) (start (grid p) a s)
      exact IOSafety.source_safe M dot p _ (domain.launch live) bounds (bound a member s)
    · intro next _
      exact ih (fun a ha=>members a (by simp [ha])) next

end
end VeriTile.Bench.Repairs.AttentionKernel.IOGridSafety
