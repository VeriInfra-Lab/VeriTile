import Lean

/-- Read existing module headers without rebuilding reviewed modules. -/
def main (files : List String) : IO Unit := do
  for file in files do
    let (data, _) ← Lean.readModuleData file
    IO.println (Lean.Json.compress (Lean.Json.mkObj [
      ("path", Lean.toJson file),
      ("imports", Lean.toJson (data.imports.toList.map fun i => i.module.toString))]))
