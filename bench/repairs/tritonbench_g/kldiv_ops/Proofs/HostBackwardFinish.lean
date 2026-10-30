import bench.repairs.tritonbench_g.kldiv_ops.Proofs.HostForward

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
noncomputable section
set_option maxHeartbeats 6000000

structure Modified (dst : Tensor) (before after : State) : Prop where
  live : ∀ b,b∈before.live → b∈after.live
  frame : ∀ b,b∈before.live → ∀ p,b.contains p → ¬Written dst p → after.heap p=before.heap p

theorem Preserves.modified {s t : State} (h : Preserves s t) (dst : Tensor) : Modified dst s t :=
  ⟨h.live,fun b hb p hp _=>h.frame b hb p hp⟩

theorem Modified.trans {dst : Tensor} {s t u : State} (a : Modified dst s t) (b : Modified dst t u) :
    Modified dst s u := by
  refine ⟨fun x hx=>b.live x (a.live x hx),?_⟩
  intro x hx p hp outside
  exact (b.frame x (a.live x hx) p hp outside).trans (a.frame x hx p hp outside)

theorem Copied.modified {Word : Type} {codec : Codec Word} {neg : Word→Word}
    {dst src returned : Tensor} {s t : State} (h : Copied codec neg dst src returned s t) : Modified dst s t :=
  ⟨by intro b hb; simpa only [h.live] using hb,h.frame⟩

theorem written_inside (dst : Tensor) (p : Nat) (h : Written dst p) : dst.backing.contains p := by
  obtain ⟨i,hi,lo,up⟩ := h
  have inside := byte_inside dst i hi ⟨p-dst.address i,by omega⟩
  have eq : dst.address i+(p-dst.address i)=p := by omega
  simpa only [eq] using inside

theorem disjoint_unwritten (dst : Tensor) (backing : Backing) (apart : dst.backing.disjoint backing)
    (p : Nat) (hp : backing.contains p) : ¬Written dst p := by
  intro h
  have hd := written_inside dst p h
  simp only [Backing.disjoint,Backing.contains] at *
  omega

theorem Copied.separated_frame {Word : Type} {codec : Codec Word} {neg : Word→Word}
    {dst src returned : Tensor} {s t : State} (h : Copied codec neg dst src returned s t)
    (b : Backing) (live : b∈s.live) (apart : dst.backing.disjoint b) : Frame s.heap t.heap b := by
  intro p hp
  exact h.frame b live p hp (disjoint_unwritten dst b apart p hp)

structure Product {Word : Type} (codec : Codec Word) (num : Numerics Word)
    (a b out : Tensor) (before after : State) : Prop where
  shape : broadcastShape a.shape b.shape=some out.shape
  allocated : Allocated out.shape a.device out before after
  values : ∀ i,i<out.numel → value codec num.neg after.heap out i=num.multiply
    (value codec num.neg before.heap a (broadcastIndex out.shape a.shape i))
    (value codec num.neg before.heap b (broadcastIndex out.shape b.shape i))

structure BackwardFinished {Word : Type} (codec : Codec Word) (num : Numerics Word)
    (grad dst work returned : Tensor) (before after : State) where
  (copied compared : State)
  one : Tensor
  isOne : Bool
  copy : Copied codec num.neg dst work dst before copied
  scalar : Allocated [] grad.device one copied compared
  scalarValue : value codec num.neg compared.heap one 0=num.one
  test : isOne=tensorEqual codec num compared.heap grad one
  gradValues : ∀ i,i<grad.numel → value codec num.neg compared.heap grad i=value codec num.neg before.heap grad i
  comparedValues : ∀ i,i<dst.numel → value codec num.neg compared.heap dst i=value codec num.neg before.heap work i
  destinationValues : ∀ i,i<dst.numel → value codec num.neg after.heap dst i=value codec num.neg before.heap work i
  returnCase : if isOne then returned=dst ∧ after=compared else Product codec num dst grad returned compared after
  callerFrame : Modified dst before after
  live : returned.backing∈after.live

theorem backwardFinish_correct {Word : Type} (codec : Codec Word) (num : Numerics Word)
    (api : Services Word) (lib : Library codec num api) (grad dst work returned : Tensor) (s t : State)
    (glive : grad.backing∈s.live) (dlive : dst.backing∈s.live) (wlive : work.backing∈s.live)
    (shape : dst.shape=work.shape) (apart : dst.backing.disjoint grad.backing)
    (call : backwardFinish api grad dst work s=.ok returned t) :
    Nonempty (BackwardFinished codec num grad dst work returned s t) := by
  obtain ⟨copiedResult,copied,hcopy,h⟩ := bind_success call
  have copy := lib.copy dst work s copiedResult copied dlive wlive shape hcopy
  have grlive : grad.backing∈copied.live := by rw [copy.live]; exact glive
  have dstlive : dst.backing∈copied.live := by rw [copy.live]; exact dlive
  obtain ⟨one,compared,hone,h⟩ := bind_success h
  have scalar := lib.one grad.device copied one compared hone
  obtain ⟨isOne,checked,hequal,h⟩ := bind_success h
  have eq := lib.equal grad one compared isOne checked
    (scalar.1.preserved.live _ grlive) scalar.1.live hequal
  have checkedEq := eq.1
  subst checked
  have gv : ∀ i,i<grad.numel → value codec num.neg compared.heap grad i=value codec num.neg s.heap grad i := by
    intro i hi
    rw [preserved_value codec num.neg scalar.1.preserved grad grlive i hi]
    exact Storage.preserves_value codec num.neg s.heap copied.heap grad (copy.separated_frame _ glive apart) i hi
  have dv : ∀ i,i<dst.numel → value codec num.neg compared.heap dst i=value codec num.neg s.heap work i := by
    intro i hi
    rw [preserved_value codec num.neg scalar.1.preserved dst dstlive i hi]
    exact copy.values i hi
  have cp : Copied codec num.neg dst work dst s copied := by simpa only [copy.identity] using copy
  cases isOne with
  | «true» =>
    simp only [Bool.true_eq,ite_true,pure,Result.ok.injEq] at h
    rcases h with ⟨rfl,rfl⟩
    exact ⟨⟨copied,compared,one,Bool.true,cp,scalar.1,scalar.2,eq.2,gv,dv,dv,
      ⟨rfl,rfl⟩,copy.modified.trans (scalar.1.preserved.modified dst),scalar.1.preserved.live _ dstlive⟩⟩
  | «false» =>
    have hp : api.multiply dst grad compared=.ok returned t := h
    have product := lib.multiply dst grad compared returned t
      (scalar.1.preserved.live _ dstlive) (scalar.1.preserved.live _ grlive) hp
    refine ⟨⟨copied,compared,one,Bool.false,cp,scalar.1,scalar.2,eq.2,gv,dv,?_,
      ⟨product.1,product.2.1,product.2.2⟩,
      copy.modified.trans ((scalar.1.preserved.trans product.2.1.preserved).modified dst),product.2.1.live⟩⟩
    intro i hi
    rw [preserved_value codec num.neg product.2.1.preserved dst (scalar.1.preserved.live _ dstlive) i hi]
    exact dv i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host
