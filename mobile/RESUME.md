# Supported resume statements

Use these as engineering statements; adapt them to your actual contribution and
the validation receipts linked from the app README.

- Built a React/Capacitor Android/iOS companion integrating **2 AI project
  backends**, with persistent research notes, source inspection, match reports,
  and visible agent tool traces.
- Implemented an MCP research-agent interface bounded to **4 model rounds and
  6 tool calls**, with excerpt fingerprints and checked source references;
  validated behavior through **36 agent/MCP tests**, including SDK stdio calls.
- Passed **51 frontend unit checks and 22 browser checks** across two mobile
  viewport sizes, covering response contracts, failure states, approval pauses,
  and saved-item persistence; compiled a signed Android debug APK and an unsigned
  iOS simulator app.

Evidence: [app checks](demo/validation.json), [agent/MCP checks](../doc/research-agent-validation.json),
[native builds](demo/native-validation.json), and [passing native CI](https://github.com/joses2017smjh/MetaNavT/actions/runs/37754858391).
The 22 browser checks comprise 18 feature/error cases and 4 actual-server
integration cases, all in Chromium mobile viewports. Do not claim App Store
distribution, physical-device testing, real-match accuracy, or robotics gains
from these synthetic examples. A working provider connector is distinct from a
measured live-model evaluation; the demo uses a labeled scripted policy.

For robotics/AI applications, emphasize research provenance, remote agent
inspection, bounded execution, reproducible contracts, and failure handling.
