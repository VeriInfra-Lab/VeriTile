import bench.tritonbench_g.flash_decode2_phi.Proofs.Reference
import bench.tritonbench_g.flash_decode2_phi.Proofs.Coordinates

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.KernelInterface
open VeriTile.Triton FP.Structural FlashDecode2
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000

/-- Raw scalar launch geometry for the unmasked Llama source. -/
structure Geometry where
  B : Nat
  H : Nat
  S : Nat
  D : Nat
  Q : Nat
  positiveQ : 0<Q

def steps [Inhabited α] (g : KernelInterface.Geometry) (s : State α) (row : Nat) : Nat :=
  ((s.mem "L" (row/g.H)).read .nat+g.Q-1)/g.Q

def LengthBounds [Inhabited α] (g : KernelInterface.Geometry) (s : State α) : Prop :=
  ∀ row,row<g.B*g.H → 0<(s.mem "L" (row/g.H)).read .nat ∧
    (s.mem "L" (row/g.H)).read .nat ≤ g.S*g.Q

/-- Input observations on the raw E/M regions; all other cells, including
length metadata and arbitrary surrounding storage, retain their entry values. -/
def inputState (data : RegionName → Nat → α) (s : State α) : State α :=
  {s with mem := fun r i => if r="E" ∨ r="M" then .mk .real (data r i) else s.mem r i}

theorem count_bound [Inhabited α] (g : KernelInterface.Geometry) (s : State α)
    (bounds : LengthBounds g s) (row : Nat) (inside : row<g.B*g.H) :
    count (Grid.setup g.B g.H row s) g.Q=steps g s row ∧
      0<steps g s row ∧ steps g s row ≤ g.S := by
  have h := bounds row inside
  refine ⟨count_eq _ _ _ rfl h.1,ceil_positive _ _ h.1 g.positiveQ,
    ceil_capacity _ _ _ g.positiveQ h.2⟩

theorem typed [Inhabited α] (A : Algebra α) (g : KernelInterface.Geometry)
    (order : List Nat) (coverage : order.Perm (List.range (g.B*g.H))) (before : State α) :
    ∃ after,Grid.source A Bool.false g.B g.H g.S g.D g.Q g.D order before=some after ∧
      (∀ row (k : TileIndex [g.D]),row<g.B*g.H → after.mem "O" (row*g.D+lane k)=
        .mk .real (Grid.value A before Bool.false g.B g.H g.S g.D g.Q g.D row k)) ∧
      (∀ r i,r≠("O" : RegionName) ∨ g.B*g.H*g.D ≤ i → after.mem r i=before.mem r i) := by
  obtain ⟨after,run,memory⟩ := Grid.source_complete A Bool.false g.B g.H g.S g.D g.Q g.D order before
  refine ⟨after,run,?_,?_⟩
  · intro row k inside
    rw [memory]
    exact Grid.grid_readback A Bool.false g.B g.H g.S g.D g.Q g.D (Or.inr rfl)
      order before row (coverage.mem_iff.mpr (List.mem_range.mpr inside)) k k.1.isLt
  · intro r i outside
    rw [memory]
    apply Grid.grid_frame A Bool.false g.B g.H g.S g.D g.Q g.D (Or.inr rfl) order before r i
    rcases outside with outside|outside
    · exact Or.inl outside
    · right
      intro row member d hd same
      have inside := List.mem_range.mp (coverage.mem_iff.mp member)
      have bound : row*g.D+d<g.B*g.H*g.D := by nlinarith
      omega

def realEntry (data : RegionName → Nat → ℝ) (s : State (Option ℝ)) : State (Option ℝ) :=
  inputState (fun r i => some (data r i)) s

def realTarget (g : KernelInterface.Geometry) (data : RegionName → Nat → ℝ)
    (s : State (Option ℝ)) (row d : Nat) : ℝ :=
  Reference.target (fun j => data "E" (row*g.S+j))
    (fun j d => data "M" (row*g.S*g.D+j*g.D+d)) (steps g s row) d

theorem real_value (g : KernelInterface.Geometry) (data : RegionName → Nat → ℝ) (before : State (Option ℝ))
    (bounds : LengthBounds g before) (row : Nat) (inside : row<g.B*g.H) (k : TileIndex [g.D]) :
    Grid.value Reference.ideal (realEntry data before) Bool.false g.B g.H g.S g.D g.Q g.D row k=
      some (realTarget g data before row (lane k)) := by
  let s := Grid.setup g.B g.H row (realEntry data before)
  let x := fun j => data "E" (row*g.S+j)
  let v := fun j (k : TileIndex [g.D]) => data "M" (row*g.S*g.D+j*g.D+lane k)
  have hb : LengthBounds g (realEntry data before) := by
    simpa only [LengthBounds,realEntry,inputState,show ¬(("L" : RegionName)="E" ∨ "L"="M") by decide,if_false] using bounds
  have counted := count_bound g (realEntry data before) hb row inside
  have hx : ∀ j,j<count s g.Q → logvalue Reference.ideal s g.H g.S j=some (x j) := by
    intro j _
    simp only [s,logvalue,row_log_offset]
    simp [Reference.ideal,realEntry,inputState,Grid.setup,Cell.read_same,x]
  have hv : ∀ j,j<count s g.Q → ∀ k,
      FlashDecode2.value Reference.ideal s Bool.false g.H g.S g.D j k=some (v j k) := by
    intro j _ k
    simp only [s,FlashDecode2.value,row_mid_offset]
    simp [active,Reference.ideal,realEntry,inputState,Grid.setup,Cell.read_same,v]
  have result := Reference.source_correct s Bool.false g.H g.S g.D g.Q g.D x v
    (counted.1.symm ▸ counted.2.1) hx hv k
  change output Reference.ideal s _ _ _ _ _ _ k=_
  rw [result]
  change some (Reference.target x v (count s g.Q) k)=_
  rw [counted.1]
  rfl

#axiomsClean typed
#axiomsClean real_value
#specNonCircular realTarget avoiding [Grid.source,FP.Structural.exec,values,Reference.recurrence]
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.KernelInterface
