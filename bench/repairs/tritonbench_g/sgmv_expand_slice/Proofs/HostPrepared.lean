import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostSnapshots
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostMetadata
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.PreparedMemory

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.HostPrepared
open HostSnapshots

def scalarBytes : ScalarFormat → Nat
  | .half | .bfloat => 2 | .single => 4
def integerBytes : MetadataWidth → Nat
  | .w32 => 4 | .w64 => 8

/-- Successful source shape/dtype checks. Caller aliases and lazy-negative
views are unrestricted; these are finite logical contents at call entry. -/
structure Domain {α : Host.Role → Type} (abi : SourceABI) (shape : Shape)
    (batches : Nat) (entry : (role : Host.Role) → View α role) : Prop where
  inputShape : (entry .input).dimensions = [shape.tokens,shape.inner]
  weightShape : (entry .weight).dimensions = [shape.adapters,shape.columns,shape.inner]
  outputShape : (entry .output).dimensions = [shape.outputRows,shape.outputColumns]
  startsShape : (entry .starts).dimensions = [batches]
  lengthsShape : (entry .lengths).dimensions = [(entry .lengths).values.length]
  adaptersShape : (entry .adapters).dimensions = [batches]
  lengthsEnough : batches ≤ (entry .lengths).values.length
  inputWidth : (entry .input).bytesPerElement = scalarBytes abi.input
  weightWidth : (entry .weight).bytesPerElement = scalarBytes abi.weight.scalar
  outputWidth : (entry .output).bytesPerElement = scalarBytes abi.output
  startsWidth : (entry .starts).bytesPerElement = integerBytes abi.startsWidth
  lengthsWidth : (entry .lengths).bytesPerElement = integerBytes abi.lengthsWidth
  adaptersWidth : (entry .adapters).bytesPerElement = integerBytes abi.adaptersWidth
  capacity : shape.slice+shape.columns ≤ shape.outputColumns

def strides {α : Host.Role → Type} {entry : (role : Host.Role) → View α role} {caller : List Window}
    (p : Prepared entry caller) : PreparedMemory.Strides :=
  ⟨p.x.stride 0,p.x.stride 1,p.w.stride 0,p.w.stride 1,p.w.stride 2,p.y.stride 0,p.y.stride 1⟩

theorem layouts {α : Host.Role → Type} (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View α role) (caller : List Window)
    (domain : Domain abi shape batches entry) (p : Prepared entry caller) :
    GridStride.Contiguous shape.tokens shape.inner (strides p).inputRow (strides p).inputInner ∧
    SnapshotLayout.WeightContiguous shape.adapters shape.columns shape.inner
      (strides p).weightAdapter (strides p).weightColumn (strides p).weightInner ∧
    GridStride.Contiguous shape.outputRows shape.outputColumns (strides p).outputRow (strides p).outputColumn :=
  ⟨p.x.matrixLayout _ _ domain.inputShape,p.w.weightLayout _ _ _ domain.weightShape,
    p.y.matrixLayout _ _ domain.outputShape⟩

theorem metadata_lengths {α : Host.Role → Type} (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View α role) (caller : List Window)
    (domain : Domain abi shape batches entry) (p : Prepared entry caller) :
    p.starts.values.length = batches ∧ batches ≤ p.lengths.values.length ∧ p.adapters.values.length = batches := by
  have hs := (entry .starts).extent
  have ha := (entry .adapters).extent
  simp only [domain.startsShape,domain.adaptersShape,List.prod_cons,List.prod_nil,Nat.mul_one] at hs ha
  exact ⟨by simpa [p.starts.preserved,payloadLength] using hs,
    by simpa [p.lengths.preserved] using domain.lengthsEnough,
    by simpa [p.adapters.preserved,payloadLength] using ha⟩

def matrix {β : Type} [Inhabited β] (columns : Nat) (values : List β) (r c : Nat) : β :=
  (values[r*columns+c]?).getD default
def weight {β : Type} [Inhabited β] (columns inner : Nat) (values : List β) (a c k : Nat) : β :=
  (values[(a*columns+c)*inner+k]?).getD default

/-- The local tolist API preserves the signed logical values. It can fail;
the API contains no validation or successful-kernel condition. -/
structure ToListAPI where
  call : (role : Host.Role) → (values : List Int) → Except Unit {out : List Int // out = values}

structure Lists {α : Host.Role → Type} {entry : (role : Host.Role) → View α role} {caller : List Window}
    (p : Prepared entry caller) where
  starts : {out : List Int // out = p.starts.values}
  lengths : {out : List Int // out = p.lengths.values}
  adapters : {out : List Int // out = p.adapters.values}

def toLists {α : Host.Role → Type} {entry : (role : Host.Role) → View α role} {caller : List Window}
    (api : ToListAPI) (p : Prepared entry caller) : Except Host.Stage (Lists p) := do
  let starts ← (api.call .starts p.starts.values).mapError (fun _ => .toList .starts)
  let lengths ← (api.call .lengths p.lengths.values).mapError (fun _ => .toList .lengths)
  let adapters ← (api.call .adapters p.adapters.values).mapError (fun _ => .toList .adapters)
  pure ⟨starts,lengths,adapters⟩

def raw {α : Host.Role → Type} {entry : (role : Host.Role) → View α role} {caller : List Window}
    {p : Prepared entry caller} (batches : Nat) (lists : Lists p) : List RawSegment :=
  HostMetadata.rawSegments batches lists.starts.val lists.lengths.val lists.adapters.val

theorem raw_entry {α : Host.Role → Type} {entry : (role : Host.Role) → View α role} {caller : List Window}
    {p : Prepared entry caller} (batches : Nat) (lists : Lists p) :
    raw batches lists = HostMetadata.rawSegments batches
      (entry .starts).values (entry .lengths).values (entry .adapters).values := by
  simp only [raw,lists.starts.property,lists.lengths.property,lists.adapters.property,
    p.starts.preserved,p.lengths.preserved,p.adapters.preserved]

theorem metadata_same_snapshot {α : Host.Role → Type} (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View α role) (caller : List Window)
    (domain : Domain abi shape batches entry) (p : Prepared entry caller) (lists : Lists p) :
    (raw batches lists).map RawSegment.start = p.starts.values ∧
    (raw batches lists).map RawSegment.length ++ lists.lengths.val.drop batches = p.lengths.values ∧
    (raw batches lists).map RawSegment.adapter = p.adapters.values := by
  obtain ⟨hs,hl,ha⟩ := metadata_lengths abi shape batches entry caller domain p
  simp only [raw,lists.starts.property,lists.lengths.property,lists.adapters.property]
  have h := HostMetadata.projections batches p.starts.values p.lengths.values p.adapters.values hs hl ha
  exact ⟨h.1,HostMetadata.full_lengths _ _ _ _ hs hl ha,h.2.2⟩

#axiomsClean layouts
#axiomsClean metadata_lengths
#axiomsClean raw_entry
#axiomsClean metadata_same_snapshot
end VeriTile.Bench.Repairs.SgmvExpandSlice.HostPrepared
