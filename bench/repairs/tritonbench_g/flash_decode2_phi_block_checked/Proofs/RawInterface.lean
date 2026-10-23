import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.KernelNative
import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Grid

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.Raw
open VeriTile.Triton FP.Structural FlashDecode2
noncomputable section

/-- Raw masked stage-two launch: complete source execution, every selected
output lane and every exterior typed memory cell. -/
theorem complete {α : Type} [Inhabited α] (A : Algebra α) (B H S D Q BD : Nat)
    (order : List Nat) (all : order.Perm (List.range (B*H))) (before : State α) :
    ∃ after,Grid.source A Bool.true B H S D Q BD order before=some after ∧
      (∀ row (k : TileIndex [BD]),row< B*H → lane k< D →
        after.mem "O" (row*D+lane k)=.mk .real (Grid.value A before Bool.true B H S D Q BD row k)) ∧
      (∀ r i,r ≠ ("O" : RegionName) ∨ B*H*D ≤ i → after.mem r i=before.mem r i) := by
  obtain ⟨after,executed,memory⟩ := Grid.source_complete A Bool.true B H S D Q BD order before
  refine ⟨after,executed,?_,?_⟩
  · intro row k hr hk
    rw [memory]
    exact Grid.grid_readback A _ _ _ _ _ _ _ (Or.inl rfl) order before row
      (all.mem_iff.mpr (List.mem_range.mpr hr)) k hk
  · intro r i outside
    rw [memory]
    apply Grid.grid_frame A _ _ _ _ _ _ _ (Or.inl rfl) order before r i
    rcases outside with outside | outside
    · exact Or.inl outside
    · right
      intro row member d hd same
      have hr := List.mem_range.mp (all.mem_iff.mp member)
      have bound : row*D+d < B*H*D := by nlinarith
      omega

def logs (s : State Native.Term) (S row j : Nat) : Native.Term :=
  .load ((s.mem "E" (row*S+j)).read .real)
def mids (s : State Native.Term) (S D row j d : Nat) : Native.Term :=
  .load (if d< D then (s.mem "M" (row*S*D+j*D+d)).read .real
    else .literal (some .fp32) .real 0)

def nativeTerm (s : State Native.Term) (B H S D Q row d : Nat) : Native.Term :=
  let a := KernelNative.scan (logs s S row) (mids s S D row) (count (Grid.setup B H row s) Q)
  KernelNative.op .div (a.numerator d) a.denominator

theorem recurrence (s : State Native.Term) (B H S D BD row n : Nat) :
    let entry := Grid.setup B H row s
    let a := KernelNative.scan (logs s S row) (mids s S D row) n
    (values Native.algebra entry Bool.true H S D BD n).m=a.maximum ∧
    (values Native.algebra entry Bool.true H S D BD n).z=a.denominator ∧
    ∀ k,(values Native.algebra entry Bool.true H S D BD n).a k=a.numerator (lane k) := by
  have hx (j : Nat) : logvalue Native.algebra (Grid.setup B H row s) H S j=logs s S row j := by
    simp only [logvalue,row_log_offset]
    simp [Native.algebra,Grid.setup,logs]
  have hv (j : Nat) (k : TileIndex [BD]) :
      FlashDecode2.value Native.algebra (Grid.setup B H row s) Bool.true H S D j k=mids s S D row j (lane k) := by
    simp only [FlashDecode2.value,row_mid_offset]
    simp [Native.algebra,Grid.setup,mids,active,zero]
  induction n with
  | zero => simp [values,FlashDecode2.initial,zero,Native.algebra,KernelNative.scan,KernelNative.initial]
  | succ n ih =>
    dsimp only at ih ⊢
    simp only [values,next,ih.1,ih.2.1,ih.2.2,hx,hv]
    simp only [maximum,exponential,sub,FlashDecode2.mul,add,Native.algebra,
      KernelNative.scan,KernelNative.merge,KernelNative.op,KernelNative.exp,and_self,implies_true]

theorem native_value (s : State Native.Term) (B H S D Q BD row : Nat) (k : TileIndex [BD]) :
    Grid.value Native.algebra s Bool.true B H S D Q BD row k=nativeTerm s B H S D Q row (lane k) := by
  have recur := recurrence s B H S D BD row (count (Grid.setup B H row s) Q)
  simp only [Grid.value,output]
  rw [recur.2.2 k,recur.2.1]
  rfl

#axiomsClean complete
#axiomsClean native_value
#specNonCircular nativeTerm avoiding [Grid.source,FP.Structural.exec,values]
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.Raw
