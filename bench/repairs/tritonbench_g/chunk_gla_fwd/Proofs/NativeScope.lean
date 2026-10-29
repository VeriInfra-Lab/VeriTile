import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeGrid

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeScope
open VeriTile Triton FP.Structural
open FPStorage (Memory payload)
open FPStages (Stage)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000

/-- These observers are proof-side memory observations. Actual execution
keeps the original typed cells; the separate tag theorem recovers each
public output channel's exact Structural dtype. -/
def producerObserve (r : RegionName) (cell : Cell α) : α :=
  if r="attention" then payload cell else cell.read .real

def consumerObserve (r : RegionName) (cell : Cell α) : α :=
  if r="o" then payload cell else if r="h" then cell.read .fp32 else cell.read .real

def inputScope (c : Host.Dimensions) (r : RegionName) (off : Nat) : Prop :=
  (r="q" ∨ r="k" ∨ r="g") ∧ off<c.heads*c.T*c.K

def producerScope (c : Host.Dimensions) (scratch : Bool) (r : RegionName) (off : Nat) : Prop :=
  inputScope c r off ∨ (r="attention" ∧ off<Host.attentionSize c) ∨
    (scratch=Bool.true ∧ r="scratch" ∧ off<Host.scratchSize c)

def consumerMemory (s : State α) : Memory α := fun r off=>consumerObserve r (s.mem r off)

theorem inter_ports (c : Host.Dimensions) (scale : ℝ) (scratch : Bool) :
    NativeGrid.Covers (producerScope c scratch) (.inter c.producer scale) ∧
    NativeGrid.ReadCompatible (producerObserve (α:=α)) (.inter c.producer scale) ∧
    NativeProgram.WriteCompatible (producerObserve (α:=α)) (.inter c.producer scale) := by
  refine ⟨?_,?_,?_⟩
  · intro port member off inside
    simp only [NativeLaunch.inputExtents,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl <;> exact Or.inl ⟨by simp,inside⟩
  · simp [NativeGrid.ReadCompatible,NativeLaunch.inputExtents,producerObserve]
  · simp [NativeProgram.WriteCompatible,producerObserve,payload]

theorem intra_ports (c : Host.Dimensions) (scale : ℝ) (scratch : Bool) :
    NativeGrid.Covers (producerScope c scratch) (.intra c.producer c.intraBK scale) ∧
    NativeGrid.ReadCompatible (producerObserve (α:=α)) (.intra c.producer c.intraBK scale) ∧
    NativeProgram.WriteCompatible (producerObserve (α:=α)) (.intra c.producer c.intraBK scale) := by
  refine ⟨?_,?_,?_⟩
  · exact (inter_ports (α:=α) c scale scratch).1
  · simp [NativeGrid.ReadCompatible,NativeLaunch.inputExtents,producerObserve]
  · simp [NativeProgram.WriteCompatible,producerObserve,payload]

theorem split_ports (c : Host.Dimensions) (scale : ℝ) :
    NativeGrid.Covers (producerScope c Bool.true) (.split c.split scale) ∧
    NativeGrid.ReadCompatible (producerObserve (α:=α)) (.split c.split scale) ∧
    NativeProgram.WriteCompatible (producerObserve (α:=α)) (.split c.split scale) := by
  refine ⟨?_,?_,?_⟩
  · exact (inter_ports (α:=α) c scale Bool.true).1
  · simp [NativeGrid.ReadCompatible,NativeLaunch.inputExtents,producerObserve]
  · simp [NativeProgram.WriteCompatible,producerObserve,Cell.read_same]

theorem merge_ports (c : Host.Dimensions) :
    NativeGrid.Covers (producerScope c Bool.true) (.merge c.split) ∧
    NativeGrid.ReadCompatible (producerObserve (α:=α)) (.merge c.split) ∧
    NativeProgram.WriteCompatible (producerObserve (α:=α)) (.merge c.split) := by
  refine ⟨?_,?_,?_⟩
  · intro port member off inside
    simp only [NativeLaunch.inputExtents,List.mem_singleton] at member
    subst port
    exact Or.inr (Or.inr ⟨rfl,rfl,inside⟩)
  · simp [NativeGrid.ReadCompatible,NativeLaunch.inputExtents,producerObserve]
  · simp [NativeProgram.WriteCompatible,producerObserve,payload]

theorem output_ports (c : Host.Dimensions) (scale : ℝ) :
    NativeGrid.Covers (fun _ _=>True) (.output c.consumer scale) ∧
    NativeGrid.ReadCompatible (consumerObserve (α:=α)) (.output c.consumer scale) ∧
    NativeProgram.WriteCompatible (consumerObserve (α:=α)) (.output c.consumer scale) := by
  refine ⟨fun _ _ _ _=>True.intro,?_,?_⟩
  · simp [NativeGrid.ReadCompatible,NativeLaunch.inputExtents,consumerObserve]
  · simp [NativeProgram.WriteCompatible,consumerObserve,payload]

theorem fill_reference (before after : RegionName → Nat → Prop)
    (observe : RegionName → Cell α → α) (M : Algebra α) (region : RegionName) (count : Nat)
    (included : ∀ r off,after r off → before r off ∨ (r=region ∧ off<count))
    (observed : observe region (.mk .real (M.literal (some .fp32) .real 0))=M.literal (some .fp32) .real 0)
    (s : State α) (m : Memory α) (hm : ∀ r off,before r off → observe r (s.mem r off)=m r off) :
    ∀ r off,after r off → observe r (((NativeLaunch.zero M region count).apply s).mem r off)=
      Host.zeroRange M m region count r off := by
  intro r off inside
  simp only [NativeLaunch.zero,FiniteFill.apply,FiniteFill.Contains,Nat.zero_add,Nat.zero_le,true_and,Host.zeroRange]
  by_cases active : r=region ∧ off<count
  · rw [if_pos active,if_pos active]
    simpa only [active.1] using observed
  · rw [if_neg active,if_neg active]
    rcases included r off inside with old|fresh
    · exact hm r off old
    · exact (active fresh).elim

theorem fill_frame (M : Algebra α) (region : RegionName) (count : Nat) (s : State α)
    (r : RegionName) (off : Nat) (outside : r≠region ∨ count≤off) :
    ((NativeLaunch.zero M region count).apply s).mem r off=s.mem r off := by
  apply FiniteFill.apply_mem_outside
  simp only [FiniteFill.Contains,NativeLaunch.zero,Nat.zero_add,Nat.zero_le,true_and]
  rcases outside with other|beyond
  · exact fun h=>other h.1
  · exact fun h=>Nat.not_lt_of_ge beyond h.2

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeScope
