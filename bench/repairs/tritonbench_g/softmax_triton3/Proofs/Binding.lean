import bench.repairs.tritonbench_g.softmax_triton3.Proofs.Contract
import bench.repairs.tritonbench_g.softmax_triton3.Proofs.Model
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.InputBinding
namespace VeriTile.Bench.Repairs.SoftmaxTriton3.Contiguous
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
noncomputable section

abbrev Object := SoftmaxTriton1.Contiguous.Object

def descriptor (x : Object) : Tensor where
  objectId := x.tensor.object
  region := x.region
  rows := rows x.metadata
  cols := cols x.metadata
  offset := x.tensor.storageOffset
  rowStride := cols x.metadata
  colStride := 1
  device := x.metadata.device.index
  negative := x.metadata.isNeg

def objectValue {S C V} (a : Access S C V) (neg : V → V)
    (s : S) (x : Object) (r c : Nat) : V :=
  let v := a.read s x.region (x.tensor.physicalIndex (r*cols x.metadata+c))
  if x.metadata.isNeg then neg v else v

theorem descriptor_read {S C V} (a : Access S C V) (neg : V → V)
    (s : S) (x : Object) (h : TensorContract x.metadata)
    (r c : Nat) (hr : r<rows x.metadata) (hc : c<cols x.metadata) :
    a.read s x.region ((descriptor x).index r c)=objectValue a neg s x r c := by
  have hi : r*cols x.metadata+c<x.metadata.numel := by
    rw [numel_eq _ h.rank]
    exact VeriTile.Bench.Repairs.SquareMatrix.flat_lt _ _ _ _ hr hc
  simp [descriptor,Tensor.index,objectValue,h.noNeg,x.laws.contiguous h.contiguous _ hi,Nat.add_assoc]

theorem descriptor_injective (x : Object) : (descriptor x).Injective := by
  intro r c r' c' _ hc _ hc' eq
  change c<cols x.metadata at hc
  change c'<cols x.metadata at hc'
  simp only [Tensor.index,descriptor,Nat.mul_one] at eq
  have rowEq : r=r' := VeriTile.Bench.Repairs.SquareMatrix.row_unique _ r' c' r hc'
    (by omega) (by omega)
  exact ⟨rowEq,by subst r';omega⟩

/-- A source-state memory lens. Both public entries construct this record by
ordinary record update, so its only law is definitional. It is not a runtime,
compiler, numerical or desired-output assumption. -/
structure Memory {S C V} (a : Access S C V) where
  replace : S → (RegionName → Nat → C) → S
  cells : ∀ s cells,a.cells (replace s cells)=cells

/-- Interpret the kernel's pointer arguments as windows of the actual reached
storage state. This is an address-space interpretation, not a Python copy. -/
def rebase {S C V} {a : Access S C V} (memory : Memory a)
    (x y : Object) (mask : Option Object) (s : S) : S :=
  memory.replace s (fun r i =>
    if r="out" then a.cells s y.region (y.tensor.storageOffset+i)
    else if r="inp" then a.cells s x.region (x.tensor.storageOffset+i)
    else if r="mask" then let m := mask.getD x; a.cells s m.region (m.tensor.storageOffset+i)
    else a.cells s r i)

/-- Store projection writes only this row's active output cells. Every other
backing cell is taken from the same physical prestate. -/
def project {S C V} {a : Access S C V} (memory : Memory a)
    (y : Object) (p : Nat) (s t : S) : S :=
  memory.replace s (fun r i =>
    if r=y.region ∧ y.tensor.storageOffset+p*cols y.metadata ≤ i ∧
        i<y.tensor.storageOffset+p*cols y.metadata+cols y.metadata then
      a.cells t "out" (i-y.tensor.storageOffset)
    else a.cells s r i)

def physicalEvaluate {S C V} {a : Access S C V} (memory : Memory a)
    (evaluate : Evaluator S) (x y : Object) (mask : Option Object) (B p : Nat) (s : S) : Option S :=
  (evaluate "out" "inp" (mask.map (fun _ => ("mask" : RegionName))) (rows x.metadata) (cols x.metadata) B p
    (rebase memory x y mask s)).map (project memory y p s)

theorem project_active {S C V} {a : Access S C V} (memory : Memory a)
    (y : Object) (p c : Nat) (hc : c<cols y.metadata) (s t : S) :
    a.cells (project memory y p s t) y.region ((descriptor y).index p c)=
      a.cells t "out" (p*cols y.metadata+c) := by
  simp only [project,memory.cells,descriptor,Tensor.index,Nat.mul_one]
  rw [if_pos (by exact ⟨trivial,by omega,by omega⟩)]
  congr 2
  omega

theorem project_frame {S C V} {a : Access S C V} (memory : Memory a)
    (y : Object) (p : Nat) (s t : S) : FrameRows a (descriptor y) [p] s (project memory y p s t) := by
  intro r i miss
  have off : ¬(r=y.region ∧ y.tensor.storageOffset+p*cols y.metadata ≤ i ∧
      i<y.tensor.storageOffset+p*cols y.metadata+cols y.metadata) := by
    rintro ⟨hr,lo,hi⟩
    let c := i-(y.tensor.storageOffset+p*cols y.metadata)
    have hc : c<cols y.metadata := by dsimp [c];omega
    apply miss p (by simp) c hc
    refine ⟨hr.symm,?_⟩
    simp only [descriptor,Tensor.index,Nat.mul_one]
    dsimp [c]
    omega
  simp [project,memory.cells,off]

theorem physical_row {S C V} (a : Access S C V) (memory : Memory a)
    (evaluate : Evaluator S) (formula : Formula V) (source : Source a evaluate formula)
    (x y : Object) (mask : Option Object) (shape : y.metadata.shape=x.metadata.shape)
    (B p : Nat) (hB : 0<B) (cover : cols x.metadata≤B) (hp : p<rows x.metadata)
    (s t : S) (xs : Nat → V) (ms : Option (Nat → V))
    (input : ∀ c,c<cols x.metadata → a.read s x.region ((descriptor x).index p c)=xs c)
    (masks : ∀ m,mask=some m → ∀ c,c<cols x.metadata →
      a.read s m.region (m.tensor.storageOffset+p*cols x.metadata+c)=(ms.getD xs) c)
    (matching : mask.isSome=ms.isSome)
    (run : physicalEvaluate memory evaluate x y mask B p s=some t) :
    (∀ c,c<cols x.metadata → a.read t y.region ((descriptor y).index p c)=formula (cols x.metadata) B xs ms c) ∧
    FrameRows a (descriptor y) [p] s t := by
  cases he : evaluate "out" "inp" (mask.map (fun _ => ("mask" : RegionName))) (rows x.metadata)
      (cols x.metadata) B p (rebase memory x y mask s) with
  | none => simp [physicalEvaluate,he] at run
  | some u =>
    have eq : project memory y p s u=t := by simpa [physicalEvaluate,he] using run
    subst t
    have outcols : cols y.metadata=cols x.metadata := by simp [cols,shape]
    have sourceInput : ∀ c,c<cols x.metadata →
        a.read (rebase memory x y mask s) "inp" (p*cols x.metadata+c)=xs c := by
      intro c hc
      simpa [rebase,Access.read,memory.cells,descriptor,Tensor.index,Nat.add_assoc] using input c hc
    have sourceMasks : ∀ (r : RegionName),mask.map (fun _ => ("mask" : RegionName))=some r → ∀ c,c<cols x.metadata →
        a.read (rebase memory x y mask s) r (p*cols x.metadata+c)=
          (ms.getD (fun _ => a.read (rebase memory x y mask s) "inp" 0)) c := by
      intro r hr c hc
      cases mask with
      | none => simp at hr
      | some m =>
        have rr : r="mask" := by simpa using hr.symm
        subst r
        cases ms with
        | none => simp at matching
        | some f =>
          simpa [rebase,Access.read,memory.cells,Nat.add_assoc] using masks m rfl c hc
    have src := source.row "out" "inp" (mask.map (fun _ => ("mask" : RegionName))) (rows x.metadata)
      (cols x.metadata) B p (rebase memory x y mask s) u xs ms hB cover hp
      sourceInput sourceMasks (by simpa using matching) he
    refine ⟨?_,project_frame memory y p s u⟩
    intro c hc
    unfold Access.read
    rw [project_active memory y p c (by rwa [outcols]),outcols]
    exact src.1 c hc

#axiomsClean descriptor_read
#axiomsClean project_frame
#axiomsClean physical_row
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.Contiguous
