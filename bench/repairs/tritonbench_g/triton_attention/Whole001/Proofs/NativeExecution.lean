import bench.repairs.tritonbench_g.triton_attention.Whole001.NativeKernels
import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.MaskedMemory
namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
open Memory
noncomputable section
set_option maxHeartbeats 4000000

/-- Current-memory stage stores. All buffers use the same proven canonical
flattening; downstream stages reload their current stored contents. -/
def forward (c : Shape) (p : Ops α) (owners : Owners) (s : Memory α) (scale : α) : Memory α :=
  let ref := forwardReference c p owners (input c s) scale
  let so := writeTensor c s .o (fun i => ref.y i.1.val i.2.1.val i.2.2.val)
  let sl := writeRows c so .l (fun i => ref.l i.1.val i.2.val)
  writeRows c sl .m (fun i => ref.m i.1.val i.2.val)

def saved (c : Shape) (s : Memory α) : Saved α :=
  ⟨load c s .o,loadRow c s .l,loadRow c s .m⟩

def preprocess (c : Shape) (p : Ops α) (owners : Owners) (s : Memory α) : Memory α :=
  let ref := preprocessReference c p owners (load c s .g) (saved c s)
  let sn := writeTensor c s .newDO (fun i => ref.newDO i.1.val i.2.1.val i.2.2.val)
  writeRows c sn .delta (fun i => ref.delta i.1.val i.2.val)

def zeroDQ (c : Shape) (p : Ops α) (s : Memory α) : Memory α :=
  writeTensor c s .dq (fun _ => p.zero)

/-- Each key block touches DQ only at its later query rows and writes its own
DK/DV tile. DQ seeds are loaded from the current memory, not the initial state. -/
def backwardStep (c : Shape) (p : Ops α) (scale : α) (block : Nat) (s : Memory α) : Memory α :=
  let x := backwardInput c s
  let sq := writeTensorWhere c s .dq (fun i => block ≤ i.2.1.val / 128)
    (fun i => dqUpdate c p x scale block i.1.val i.2.1.val i.2.2.val
      (s .dq (offset c i)))
  let sk := writeTensorWhere c sq .dk (fun i => i.2.1.val / 128 = block)
    (fun i => dkReference c p x scale i.1.val i.2.1.val i.2.2.val)
  writeTensorWhere c sk .dv (fun i => i.2.1.val / 128 = block)
    (fun i => dvReference c p x scale i.1.val i.2.1.val i.2.2.val)

def backwardPrefix (c : Shape) (p : Ops α) (scale : α) (s : Memory α) : Nat → Memory α
  | 0 => zeroDQ c p s
  | n + 1 => backwardStep c p scale n (backwardPrefix c p scale s n)

def run (c : Shape) (p : Ops α) (owners : Owners) (s : Memory α) (scale : α) : Memory α :=
  backwardPrefix c p scale (preprocess c p owners (forward c p owners s scale)) c.blocks

/-- Local memory facts quantify over every cell, not merely the output lanes. -/
theorem backwardStep_other (c : Shape) (p : Ops α) (scale : α) (block : Nat) (s : Memory α)
    (r : Buffer) (o : Nat) (hq : r ≠ .dq) (hk : r ≠ .dk) (hv : r ≠ .dv) :
    backwardStep c p scale block s r o = s r o := by
  simp only [backwardStep,writeTensorWhere_other _ _ _ _ _ _ _ hv,
    writeTensorWhere_other _ _ _ _ _ _ _ hk,writeTensorWhere_other _ _ _ _ _ _ _ hq]

theorem backwardStep_input (c : Shape) (p : Ops α) (scale : α) (block : Nat) (s : Memory α) :
    backwardInput c (backwardStep c p scale block s) = backwardInput c s := by
  unfold backwardInput
  congr 1 <;> funext h i
  · funext e; exact backwardStep_other _ _ _ _ _ _ _ (by decide) (by decide) (by decide)
  · funext e; exact backwardStep_other _ _ _ _ _ _ _ (by decide) (by decide) (by decide)
  · funext e; exact backwardStep_other _ _ _ _ _ _ _ (by decide) (by decide) (by decide)
  · exact backwardStep_other _ _ _ _ _ _ _ (by decide) (by decide) (by decide)
  · funext e; exact backwardStep_other _ _ _ _ _ _ _ (by decide) (by decide) (by decide)
  · exact backwardStep_other _ _ _ _ _ _ _ (by decide) (by decide) (by decide)

theorem zeroDQ_input (c : Shape) (p : Ops α) (s : Memory α) :
    backwardInput c (zeroDQ c p s) = backwardInput c s := by
  unfold backwardInput
  congr 1 <;> funext h i
  · funext e; exact writeTensor_other _ _ _ _ _ _ (by decide)
  · funext e; exact writeTensor_other _ _ _ _ _ _ (by decide)
  · funext e; exact writeTensor_other _ _ _ _ _ _ (by decide)
  · exact writeTensor_other _ _ _ _ _ _ (by decide)
  · funext e; exact writeTensor_other _ _ _ _ _ _ (by decide)
  · exact writeTensor_other _ _ _ _ _ _ (by decide)

theorem backwardPrefix_input (c : Shape) (p : Ops α) (scale : α) (s : Memory α) (n : Nat) :
    backwardInput c (backwardPrefix c p scale s n) = backwardInput c s := by
  induction n with
  | zero => exact zeroDQ_input _ _ _
  | succ n ih => rw [backwardPrefix,backwardStep_input,ih]

theorem backwardStep_dq (c : Shape) (p : Ops α) (scale : α) (block : Nat) (s : Memory α)
    (i : Coord c) : backwardStep c p scale block s .dq (offset c i) =
      dqUpdate c p (backwardInput c s) scale block i.1.val i.2.1.val i.2.2.val (s .dq (offset c i)) := by
  simp only [backwardStep,writeTensorWhere_other _ _ _ _ _ _ _ (by decide : Buffer.dq ≠ .dv),
    writeTensorWhere_other _ _ _ _ _ _ _ (by decide : Buffer.dq ≠ .dk)]
  by_cases hi : block ≤ i.2.1.val / 128
  · exact writeTensorWhere_readback _ _ _ _ _ i hi
  · rw [writeTensorWhere_skip _ _ _ _ _ i hi]
    simp [dqUpdate,hi]

theorem dqReference_succ (c : Shape) (p : Ops α) (x : BackwardInput α) (scale : α)
    (n h i e : Nat) : dqReference c p x scale (n + 1) h i e =
      dqUpdate c p x scale n h i e (dqReference c p x scale n h i e) := by
  simp [dqReference,List.range_succ,List.foldl_append]

theorem backwardPrefix_dq (c : Shape) (p : Ops α) (scale : α) (s : Memory α)
    (n : Nat) (i : Coord c) : backwardPrefix c p scale s n .dq (offset c i) =
      dqReference c p (backwardInput c s) scale n i.1.val i.2.1.val i.2.2.val := by
  induction n with
  | zero => exact writeTensor_readback _ _ _ _ i
  | succ n ih => rw [backwardPrefix,backwardStep_dq,backwardPrefix_input,ih,dqReference_succ]

theorem backwardStep_dk (c : Shape) (p : Ops α) (scale : α) (block : Nat) (s : Memory α)
    (i : Coord c) : backwardStep c p scale block s .dk (offset c i) =
      if i.2.1.val / 128 = block then dkReference c p (backwardInput c s) scale
        i.1.val i.2.1.val i.2.2.val else s .dk (offset c i) := by
  simp only [backwardStep,writeTensorWhere_other _ _ _ _ _ _ _ (by decide : Buffer.dk ≠ .dv)]
  split_ifs with hi
  · exact writeTensorWhere_readback _ _ _ _ _ i hi
  · rw [writeTensorWhere_skip _ _ _ _ _ i hi]
    exact writeTensorWhere_other _ _ _ _ _ _ _ (by decide)

theorem backwardStep_dv (c : Shape) (p : Ops α) (scale : α) (block : Nat) (s : Memory α)
    (i : Coord c) : backwardStep c p scale block s .dv (offset c i) =
      if i.2.1.val / 128 = block then dvReference c p (backwardInput c s) scale
        i.1.val i.2.1.val i.2.2.val else s .dv (offset c i) := by
  unfold backwardStep
  split_ifs with hi
  · exact writeTensorWhere_readback _ _ _ _ _ i hi
  · rw [writeTensorWhere_skip _ _ _ _ _ i hi]
    rw [writeTensorWhere_other _ _ _ _ _ _ _ (by decide : Buffer.dv ≠ .dk)]
    exact writeTensorWhere_other _ _ _ _ _ _ _ (by decide)

theorem backwardPrefix_dk (c : Shape) (p : Ops α) (scale : α) (s : Memory α)
    (n : Nat) (i : Coord c) (hi : i.2.1.val / 128 < n) :
    backwardPrefix c p scale s n .dk (offset c i) =
      dkReference c p (backwardInput c s) scale i.1.val i.2.1.val i.2.2.val := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [backwardPrefix,backwardStep_dk,backwardPrefix_input]
    split_ifs with he
    · rfl
    · exact ih (by omega)

theorem backwardPrefix_dv (c : Shape) (p : Ops α) (scale : α) (s : Memory α)
    (n : Nat) (i : Coord c) (hi : i.2.1.val / 128 < n) :
    backwardPrefix c p scale s n .dv (offset c i) =
      dvReference c p (backwardInput c s) scale i.1.val i.2.1.val i.2.2.val := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [backwardPrefix,backwardStep_dv,backwardPrefix_input]
    split_ifs with he
    · rfl
    · exact ih (by omega)

#axiomsClean backwardPrefix_input
#axiomsClean backwardPrefix_dq
#axiomsClean backwardPrefix_dk
#axiomsClean backwardPrefix_dv
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
