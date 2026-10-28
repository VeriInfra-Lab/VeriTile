import Lean
#eval do
  IO.eprintln "stderr start"
  IO.println "stdout start"
  dbg_trace "dbg_trace start"
  IO.sleep 15000
  IO.eprintln "stderr end"
