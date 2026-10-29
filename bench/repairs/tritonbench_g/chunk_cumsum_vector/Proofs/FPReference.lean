import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.FPStorage

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPReference
open VeriTile.Triton FP.Structural FPValues
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

def read (h : Call) (cell : Cell α) : α := toFloat h.inputDType (cell.read h.inputDType.toTileDType)

/-- Reference input tile from logical Tensor values and the source's explicit
cast zero. The retained padding argument is ignored by this repaired source. -/
def loaded (M : Algebra α) (h : Call) (c : Config) (prog : Geometry.Program h)
    (logical : Nat → Nat → Nat → α) (padding : FPStorage.Padding α) (q : Nat) : Values α .real [c.BT,BS] :=
  fun i => M.cast (some .fp32) h.inputDType .real
    (if Geometry.active h c prog.feature q i.1.val i.2.1.val then
      logical prog.row (Geometry.time h c q i.1.val) (Geometry.column h prog.feature i.2.1.val)
    else M.cast (some .fp32) .real h.inputDType (M.literal (some .fp32) .real 0))

def carry (M : Algebra α) (h : Call) (c : Config) (prog : Geometry.Program h)
    (logical : Nat → Nat → Nat → α) (padding : FPStorage.Padding α) : Nat → Values α .real [BS]
  | 0 => fun _ => M.literal (some .fp32) .real 0
  | q+1 => updated M c (loaded M h c prog logical padding q) (carry M h c prog logical padding q)

def expected (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (positive : 0<c.BT)
    (prog : Geometry.Program h) (logical : Nat → Nat → Nat → α) (padding : FPStorage.Padding α)
    (time : Nat) (j : Fin BS) : Cell α :=
  FPMemory.stored M h (output M dot c (loaded M h c prog logical padding (time/c.BT))
    (carry M h c prog logical padding (time/c.BT)) (⟨time%c.BT,Nat.mod_lt _ positive⟩,j,PUnit.unit))

theorem expected_padding_irrelevant (M : Algebra α) (dot : DotBackend α)
    (h : Call) (c : Config) (positive : 0<c.BT) (prog : Geometry.Program h)
    (logical : Nat → Nat → Nat → α) (a b : FPStorage.Padding α) (time : Nat) (j : Fin BS) :
    expected M dot h c positive prog logical a time j=
      expected M dot h c positive prog logical b time j := by
  have eqLoad : ∀ q, FPReference.loaded M h c prog logical a q=
      FPReference.loaded M h c prog logical b q := fun _ => rfl
  have eqCarry : ∀ q, carry M h c prog logical a q=carry M h c prog logical b q := by
    intro q
    induction q with
    | zero => rfl
    | succ q ih => simp only [carry, ih, eqLoad]
  simp only [expected, eqCarry, eqLoad]

theorem loaded_logical (M : Algebra α) (h : Call) (p : Storage.Placement h) (c : Config)
    (prog : Geometry.Program h) (padding : FPStorage.Padding α) (heap : FPStorage.Heap α)
    (logical : Nat → Nat → Nat → α)
    (input : ∀ row time channel, row<h.B*h.H → time<h.T → channel<h.S →
      read h (heap (Storage.address p.backing.input (h.inputStrides.address h.H row time channel)))=logical row time channel)
    (q : Nat) : FPValues.loaded M h c prog.feature prog.row (FPStorage.entry h p prog padding heap) q=
      loaded M h c prog logical padding q := by
  funext i
  by_cases ha : Geometry.active h c prog.feature q i.1.val i.2.1.val
  · have ht : Geometry.time h c q i.1.val<h.T := by
      have hi := ha.1
      dsimp [Geometry.activeTime] at hi
      simp only [Geometry.time,if_pos ha.1]
      omega
    have hj : Geometry.column h prog.feature i.2.1.val<h.S := by
      have hi := ha.2
      dsimp [Geometry.activeColumn] at hi
      simp only [Geometry.column,if_pos ha.2]
      omega
    have eq := input prog.row _ _ prog.rowInside ht hj
    simpa only [FPValues.loaded,loaded,if_pos ha,FPStorage.entry,Storage.lift_input,read] using
      congrArg (M.cast (some .fp32) h.inputDType .real) eq
  · simp only [FPValues.loaded,loaded,if_neg ha]

theorem carry_logical (M : Algebra α) (h : Call) (p : Storage.Placement h) (c : Config)
    (prog : Geometry.Program h) (padding : FPStorage.Padding α) (heap : FPStorage.Heap α)
    (logical : Nat → Nat → Nat → α)
    (input : ∀ row time channel, row<h.B*h.H → time<h.T → channel<h.S →
      read h (heap (Storage.address p.backing.input (h.inputStrides.address h.H row time channel)))=logical row time channel)
    (q : Nat) : FPExecution.carry M h c prog.feature prog.row (FPStorage.entry h p prog padding heap) q=
      carry M h c prog logical padding q := by
  induction q with
  | zero => rfl
  | succ q ih => simp only [FPExecution.carry,carry,loaded_logical M h p c prog padding heap logical input q,ih]

theorem expected_logical (M : Algebra α) (dot : DotBackend α) (h : Call) (p : Storage.Placement h)
    (padding : Config → Geometry.Program h → FPStorage.Padding α) (c : Config) (positive : 0<c.BT)
    (prog : Geometry.Program h) (heap : FPStorage.Heap α) (logical : Nat → Nat → Nat → α)
    (input : ∀ row time channel, row<h.B*h.H → time<h.T → channel<h.S →
      read h (heap (Storage.address p.backing.input (h.inputStrides.address h.H row time channel)))=logical row time channel)
    (time : Nat) (j : Fin BS) : FPStorage.expected M dot h p padding c heap prog time j=
      expected M dot h c positive prog logical (padding c prog) time j := by
  simp only [FPStorage.expected,dif_pos positive,FPExecution.expected,expected,
    loaded_logical M h p c prog (padding c prog) heap logical input,
    carry_logical M h p c prog (padding c prog) heap logical input]

#axiomsClean expected_logical
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPReference
