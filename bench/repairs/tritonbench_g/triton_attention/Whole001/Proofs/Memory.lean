import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.BackwardLogical

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001

/-- Logical buffer identities. Fresh output/allocation separation is the
ordinary host-memory boundary; read-only Q/K/V may share physical storage. -/
inductive Buffer where
  | q | k | v | g | o | l | m | newDO | delta | dq | dk | dv | other
  deriving DecidableEq

abbrev Memory (α : Type) := Buffer → Nat → α

namespace Memory
variable {α ι : Type}

/-- One actual memory write, retaining every other cell. -/
def store (s : Memory α) (buffer : Buffer) (address : Nat) (value : α) : Memory α :=
  fun b o => if b = buffer ∧ o = address then value else s b o

/-- A finite launch's sequence of writes. Values are computed separately
from input tensors; this definition contains no mathematical target. -/
def scatter (buffer : Buffer) (addr : ι → Nat) (values : ι → α) :
    List ι → Memory α → Memory α
  | [],s => s
  | i :: rest,s => scatter buffer addr values rest (store s buffer (addr i) (values i))

theorem scatter_frame (buffer : Buffer) (addr : ι → Nat) (values : ι → α)
    (indices : List ι) (s : Memory α) (b : Buffer) (o : Nat)
    (h : ∀ i ∈ indices, b ≠ buffer ∨ o ≠ addr i) :
    scatter buffer addr values indices s b o = s b o := by
  induction indices generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [scatter,ih _ (fun j hj => h j (List.mem_cons_of_mem _ hj))]
    have hi := h i (List.mem_cons_self)
    simp only [store]
    split_ifs with he
    · rcases hi with hb | ho
      · exact False.elim (hb he.1)
      · exact False.elim (ho he.2)
    · rfl

theorem scatter_readback (buffer : Buffer) (addr : ι → Nat) (values : ι → α)
    (indices : List ι) (s : Memory α) (i : ι) (hi : i ∈ indices)
    (hinj : Function.Injective addr) :
    scatter buffer addr values indices s buffer (addr i) = values i := by
  induction indices generalizing s with
  | nil => simp at hi
  | cons j rest ih =>
    by_cases hr : i ∈ rest
    · exact ih _ hr
    · have hij : i = j := (List.mem_cons.mp hi).resolve_right hr
      subst j
      rw [scatter,scatter_frame]
      · simp [store]
      · intro j hj
        right
        intro he
        exact hr (hinj he ▸ hj)

noncomputable def writeTensor (c : Shape) (s : Memory α) (b : Buffer)
    (x : Coord c → α) : Memory α :=
  scatter b (offset c) x Finset.univ.toList s

theorem writeTensor_readback (c : Shape) (s : Memory α) (b : Buffer)
    (x : Coord c → α) (i : Coord c) :
    writeTensor c s b x b (offset c i) = x i := by
  apply scatter_readback
  · simp
  · exact offset_injective c

theorem writeTensor_other (c : Shape) (s : Memory α) (b : Buffer)
    (x : Coord c → α) (r : Buffer) (o : Nat) (h : r ≠ b) :
    writeTensor c s b x r o = s r o := by
  exact scatter_frame _ _ _ _ _ _ _ (fun _ _ => Or.inl h)

theorem writeTensor_outside (c : Shape) (s : Memory α) (b : Buffer)
    (x : Coord c → α) (r : Buffer) (o : Nat) (h : c.elements ≤ o) :
    writeTensor c s b x r o = s r o := by
  apply scatter_frame
  intro i _
  right
  have hi := offset_lt c i
  omega

def readTensor (c : Shape) (s : Memory α) (b : Buffer) (h : Fin c.groups) :
    Fin c.rows → Fin c.dim → α := fun i e => s b (offset c (h,i,e))

theorem read_writeTensor (c : Shape) (s : Memory α) (b : Buffer)
    (x : Coord c → α) (h : Fin c.groups) :
    readTensor c (writeTensor c s b x) b h = fun i e => x (h,i,e) := by
  funext i e
  exact writeTensor_readback c s b x (h,i,e)

theorem read_writeTensor_other (c : Shape) (s : Memory α) (b : Buffer)
    (x : Coord c → α) (r : Buffer) (h : Fin c.groups) (hr : r ≠ b) :
    readTensor c (writeTensor c s b x) r h = readTensor c s r h := by
  funext i e
  exact writeTensor_other c s b x r _ hr

/-- Rows use the same injective flattening, with their own scalar buffers. -/
def rowAddr (c : Shape) (p : Fin c.groups × Fin c.rows) := rowOffset c p.1 p.2

theorem rowAddr_injective (c : Shape) : Function.Injective (rowAddr c) := by
  intro a b h
  let zero : Fin c.dim := ⟨0,c.dim_pos⟩
  have he : offset c (a.1,a.2,zero) = offset c (b.1,b.2,zero) := by
    change rowAddr c a * c.dim + 0 = rowAddr c b * c.dim + 0
    rw [h]
  have hi := offset_injective c he
  apply Prod.ext
  · exact congrArg (fun x : Coord c => x.1) hi
  · exact congrArg (fun x : Coord c => x.2.1) hi

noncomputable def writeRows (c : Shape) (s : Memory α) (b : Buffer)
    (x : Fin c.groups × Fin c.rows → α) : Memory α :=
  scatter b (rowAddr c) x Finset.univ.toList s

def readRows (c : Shape) (s : Memory α) (b : Buffer) (h : Fin c.groups) :
    Fin c.rows → α := fun i => s b (rowOffset c h i)

theorem writeRows_readback (c : Shape) (s : Memory α) (b : Buffer)
    (x : Fin c.groups × Fin c.rows → α) (i : Fin c.groups × Fin c.rows) :
    writeRows c s b x b (rowAddr c i) = x i := by
  apply scatter_readback
  · simp
  · exact rowAddr_injective c

theorem writeRows_other (c : Shape) (s : Memory α) (b : Buffer)
    (x : Fin c.groups × Fin c.rows → α) (r : Buffer) (o : Nat) (h : r ≠ b) :
    writeRows c s b x r o = s r o := by
  exact scatter_frame _ _ _ _ _ _ _ (fun _ _ => Or.inl h)

theorem writeRows_outside (c : Shape) (s : Memory α) (b : Buffer)
    (x : Fin c.groups × Fin c.rows → α) (r : Buffer) (o : Nat)
    (h : c.groups * c.rows ≤ o) :
    writeRows c s b x r o = s r o := by
  apply scatter_frame
  intro i _
  right
  have hi := rowOffset_lt c i.1 i.2
  dsimp only [rowAddr]
  omega

theorem read_writeRows (c : Shape) (s : Memory α) (b : Buffer)
    (x : Fin c.groups × Fin c.rows → α) (h : Fin c.groups) :
    readRows c (writeRows c s b x) b h = fun i => x (h,i) := by
  funext i
  exact writeRows_readback c s b x (h,i)

theorem readRows_writeTensor_other (c : Shape) (s : Memory α) (b : Buffer)
    (x : Coord c → α) (r : Buffer) (h : Fin c.groups) (hr : r ≠ b) :
    readRows c (writeTensor c s b x) r h = readRows c s r h := by
  funext i
  exact writeTensor_other c s b x r _ hr

theorem readTensor_writeRows_other (c : Shape) (s : Memory α) (b : Buffer)
    (x : Fin c.groups × Fin c.rows → α) (r : Buffer) (h : Fin c.groups) (hr : r ≠ b) :
    readTensor c (writeRows c s b x) r h = readTensor c s r h := by
  funext i e
  exact writeRows_other c s b x r _ hr

theorem readRows_writeRows_other (c : Shape) (s : Memory α) (b : Buffer)
    (x : Fin c.groups × Fin c.rows → α) (r : Buffer) (h : Fin c.groups) (hr : r ≠ b) :
    readRows c (writeRows c s b x) r h = readRows c s r h := by
  funext i
  exact writeRows_other c s b x r _ hr

#axiomsClean scatter_frame
#axiomsClean scatter_readback
#axiomsClean writeTensor_readback
#axiomsClean writeTensor_outside
#axiomsClean writeRows_readback
#axiomsClean writeRows_outside
end Memory
end VeriTile.Bench.Repairs.TritonAttention.Whole001
