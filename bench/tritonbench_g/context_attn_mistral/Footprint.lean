import bench.tritonbench_g.context_attn_mistral.Grid
import bench.tritonbench_g.context_attn_mistral.Proofs.Execution

namespace VeriTile.Bench.Repairs.ContextAttnMistral.KernelFootprint
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

def Metadata (c : Host.Call) (s : State α) : Prop :=
  (∀ b,b<c.batches → (s.mem c.regions.start b).read .int=(c.start b : Int)) ∧
  (∀ b,b<c.batches → (s.mem c.regions.sequence b).read .int=(c.length b : Int))

theorem active_matches (c : Host.Call) (s : State α) (metadata : Metadata c s)
    (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) :
    KernelReference.active c.layout c.regions (KernelReference.atIndex s idx) i ↔ c.active idx i := by
  unfold KernelReference.active KernelReference.inputs KernelReference.atIndex KernelReference.row
  dsimp only
  rw [metadata.2 (idx.toPids 0) (Host.batch_bound c idx)]
  change ((idx.toPids 2*c.layout.BM+i.1.val : Nat) : Int)<(c.length (idx.toPids 0) : Int) ↔ _
  norm_cast

theorem offset_matches (c : Host.Call) (s : State α) (metadata : Metadata c s)
    (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) :
    KernelReference.offset c.layout c.regions (KernelReference.atIndex s idx) i=c.offset idx i := by
  unfold KernelReference.offset KernelReference.inputs KernelReference.atIndex KernelReference.row
  dsimp only
  rw [metadata.1 (idx.toPids 0) (Host.batch_bound c idx)]
  change (((c.start (idx.toPids 0) : Int)+(idx.toPids 2*c.layout.BM+i.1.val : Nat))*c.layout.outRow+
    (idx.toPids 1*c.layout.outHead : Nat)+(i.2.1.val*c.layout.outDim : Nat)).toNat=_
  simp only [←Nat.cast_add,←Nat.cast_mul,Int.toNat_natCast]
  rfl

theorem scatter_frame (p : Formula.Layout) (r : Program.Regions) (initialState s : State α)
    (v : Values α .real [p.BM,p.D]) (rg : RegionName) (off : Nat)
    (outside : ∀ i,KernelReference.active p r initialState i → rg=r.out → off=KernelReference.offset p r initialState i → False) :
    (KernelReference.scatter p r initialState s v).mem rg off=s.mem rg off := by
  classical
  apply State.masked_scatter_frame
  by_cases eq : rg=r.out
  · exact Or.inr (fun i _ on address => outside i on eq address)
  · exact Or.inl eq

theorem program_frame (c : Host.Call) (o : FPBackend.Ops α) (s : State α)
    (metadata : Metadata c s) (idx : GridIndex c.grid) :
    ∃ final,exec (FP.ProfiledRewrite.engine o.algebra)
      (KernelReference.program c.layout c.regions .fp32 (Host.scale c.layout.D))
      (KernelReference.atIndex s idx)=some final ∧
      ∀ rg off,(¬∃ i,c.active idx i ∧ rg=c.regions.out ∧ off=c.offset idx i) → final.mem rg off=s.mem rg off := by
  obtain ⟨final,source,memory⟩ := KernelReference.exec_complete c.layout c.regions (Host.scale c.layout.D) o
    (by rw [c.blockN];decide) (KernelReference.atIndex s idx)
  refine ⟨final,source,?_⟩
  intro rg off outside
  rw [memory]
  apply scatter_frame
  intro i on equal address
  exact outside ⟨i,(active_matches c s metadata idx i).mp on,equal,address.trans (offset_matches c s metadata idx i)⟩

def writes (c : Host.Call) (jobs : List (GridIndex c.grid)) (rg : RegionName) (off : Nat) : Prop :=
  ∃ idx∈jobs,∃ i,c.active idx i ∧ rg=c.regions.out ∧ off=c.offset idx i

theorem whole_frame (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid))
    (s : State α) (metadata : Metadata c s)
    (separate : c.regions.out≠c.regions.start ∧ c.regions.out≠c.regions.sequence) :
    ∃ final,KernelReference.runGrid c o jobs s=some final ∧ Metadata c final ∧
      ∀ rg off,¬writes c jobs rg off → final.mem rg off=s.mem rg off := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,metadata,fun _ _ _ => rfl⟩
  | cons idx rest ih =>
    obtain ⟨mid,source,frame⟩ := program_frame c o s metadata idx
    have untouched (rg : RegionName) (different : c.regions.out≠rg) (off : Nat) : mid.mem rg off=s.mem rg off := by
      apply frame
      rintro ⟨_,_,eq,_⟩
      exact different eq.symm
    have midmeta : Metadata c mid := by
      constructor
      · intro b hb;rw [untouched _ separate.1];exact metadata.1 b hb
      · intro b hb;rw [untouched _ separate.2];exact metadata.2 b hb
    obtain ⟨final,restSource,finalMeta,restFrame⟩ := ih mid midmeta
    refine ⟨final,by simp only [KernelReference.runGrid,source,Option.bind_some];exact restSource,finalMeta,?_⟩
    intro rg off outside
    rw [restFrame rg off (by rintro ⟨j,hj,i,on,eq,addr⟩;exact outside ⟨j,by simp [hj],i,on,eq,addr⟩)]
    apply frame
    rintro ⟨i,on,eq,addr⟩
    exact outside ⟨idx,by simp,i,on,eq,addr⟩

#axiomsClean whole_frame
end
end VeriTile.Bench.Repairs.ContextAttnMistral.KernelFootprint
