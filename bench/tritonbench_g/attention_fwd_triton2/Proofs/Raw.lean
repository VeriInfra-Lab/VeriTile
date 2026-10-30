import bench.tritonbench_g.attention_fwd_triton2.Proofs.Endpoints

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelInterface
open VeriTile Triton FP.Structural Whole001 Source Whole Machine
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def RawExterior (p : Parameters) (r : RegionName) (off : Nat) : Prop :=
  r≠p.Out ∨ ∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d<96,off≠offset p.o b h n d

def RawFrame {α : Type} (p : Parameters) (before after : State α) : Prop :=
  ∀ r off,RawExterior p r off → after.mem r off=before.mem r off

theorem exterior_outside_raw {α : Type} (p : Parameters) (before : State α) (work : List Program)
    (order : (programs (grid p)).Perm work) (r : RegionName) (off : Nat)
    (outside : RawExterior p r off) : Outside p (grid p) work before r off := by
  rcases outside with hr|ho
  · exact Or.inl hr
  · right
    intro a ha i active
    rcases a with ⟨x,y,z⟩
    have member:=(mem_programs _ _ _ _).mp (order.mem_iff.mpr ha)
    have head:=head_decode p y member.2.1
    change x*128+i.1.val<p.length ∧ i.2.1.val<96 at active
    have outside:=ho (y/p.heads) head.2.1 (y%p.heads) head.2.2
      (x*128+i.1.val) active.1 i.2.1.val active.2
    simpa [Source.address,Source.base,start,ids,offset] using outside

/-- Derive complete quoted-grid success from admission, including every
logical output and every untouched source location. No run/output premise. -/
theorem complete_raw {α : Type} [Inhabited α] (M : Algebra α) (dot : Dot α)
    (p : Parameters) (bases : Bases) (dom : Domain p bases)
    (sep : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS) (before : State α)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    ∃ after,runPrograms (fun _=>M) (fun _=>dot) p (grid p) work before=some after ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,
        ∀ d (hd : d<96),after.mem p.Out (offset p.o b h n d)=
          .mk .bf16 (answer M dot p
            (start (grid p) (n/128,b*p.heads+h,0) before) (lane n d hd))) ∧
      RawFrame p before after := by
  let storage:=Machine.storage p dom.canonical sep
  have unique : Unique p (grid p) work before := by
    intro a ha b hb i j hi hj same
    exact grid_unique _ storage before a (order.mem_iff.mpr ha) b (order.mem_iff.mpr hb) i j hi hj same
  obtain ⟨after,run,values,frame⟩:=run_correct (fun _=>M) (fun _=>dot) p
    (Whole.separated _ storage) (grid p) work before unique
  refine ⟨after,run,?_,?_⟩
  · intro b hb h hh n hn d hd
    have active : Source.active p
        (start (grid p) (n/128,b*p.heads+h,0) before) (lane n d hd) := by
      change n/128*128+n%128<p.length ∧ d<96
      exact ⟨by omega,hd⟩
    have covers:=(grid_covers p b h n d hb hh hn hd).1
    have value:=values _ (order.mem_iff.mp covers) _ active
    rw [physical_address _ before b h n d hh hd] at value
    exact value
  · intro r off outside
    exact frame r off (exterior_outside_raw p before work order r off outside)

#axiomsClean complete_raw
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelInterface
