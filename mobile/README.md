# Agent Field

An Android and iPhone companion for **MetaNavT** research retrieval and
[**Agentic Soccer**](https://github.com/joses2017smjh/Agentic-Soccer-Match-Prediction-MCP).
The same React interface is packaged in native Capacitor projects for both platforms.

## Try the features

![Agent Field feature demo](demo/feature-demo.gif)

[Watch the recording](demo/feature-demo.mp4) · [Demo screenshots](demo/)

The recording runs the actual interface in a mobile Chromium viewport. It shows:

- Search research fixtures, inspect a returned source and its byte span, and save it.
- Ask a read-only research agent to search, inspect, and cite evidence; open its source references and tool trace.
- Explore match outcome probabilities, expected goals, scorelines, and the recorded tool trace.
- Inspect an operator-review pause without executing or approving the action.
- Keep sources and reports in a persistent device library; share them through the native share sheet.
- Switch between bundled offline examples and your configured backend services.

The offline research search uses local keyword ranking over five committed **synthetic**
MetaNavT regression files. It does not run MetaNavT's neural models. Its research-agent example executes actual local
search and excerpt inspection with a scripted decision policy, explicitly labeled as
a demo. Connected agent mode uses a server-hosted Ollama tool-calling model. Soccer reports
replay the repository's actual workflow using its synthetic model and synthetic data.
Neither demo establishes real-match accuracy, robotics performance, or phone latency.

## Run locally

Use Node 20 or 22, then run from `mobile/`:

```bash
npm ci
npm run dev
```

Open `http://127.0.0.1:8081`. Demo mode works without servers or API keys.
Fonts and fixtures are bundled, so native demo mode needs no network connection.

```bash
npm test
npm run build
npm run test:e2e
```

The browser tests use Playwright Chromium. Install it with
`npx playwright install --with-deps chromium`, or set `CHROME_PATH` to a local Chrome
binary. These tests use Android and iPhone viewport sizes; native build checks are separate.

## Connect your services

Open **Connections**, choose **Connect services**, enter the two base URLs and the
soccer mobile API key, then save workspace settings. The phone sends JSON requests;
models, agents, data providers, and upstream credentials stay on your servers.

| Workspace | HTTP contract |
| --- | --- |
| Research | `GET /health`, `POST /api/retrieve/` with `{query, k: 5}` |
| Soccer | `GET /mobile/health`, `POST /mobile/predict` with `{text, mode: "live"}` and `X-API-Key` |

For a lightweight research service, follow the [MetaNavT mobile backend instructions](../README.md#android-and-ios-companion-backend).
For soccer, follow the [mobile adapter instructions](https://github.com/joses2017smjh/Agentic-Soccer-Match-Prediction-MCP#android-and-ios-companion-integration).
The soccer adapter delegates to the existing workflow; configure its gateway URL and
mobile access key on the server. The phone cannot approve a paused action.

Native connections require **HTTPS** reachable from the device. A phone's `localhost`
is the phone itself. Browser development permits HTTP loopback addresses, with explicit
backend CORS origins such as `http://127.0.0.1:8081`. No backend is deployed by this app.
Use your normal authentication boundary when exposing a research service.

The soccer token stays in memory until the app process restarts. Saved reports, source
excerpts, and service URLs use local app storage. Provider credentials are never bundled.
The optional research agent URL and its separate session token connect to `/agent/ask`.
Its server setup and MCP client configuration are documented in [the research companion guide](../doc/research-companion.md).
Only search and inspection tools are available. Citation checks establish membership
in inspected excerpts, not whether a model's conclusion follows from the evidence.

Removing a library item deletes that app entry; uninstalling clears native app storage.

## Android

Install Android Studio, JDK 21, and Android SDK 35. From `mobile/`:

```bash
npm ci
npm run build
npx cap sync android
npm run android
```

In Android Studio, run on a device or emulator. A terminal debug build is also available:

```bash
cd android
./gradlew assembleDebug
```

The output is `android/app/build/outputs/apk/debug/app-debug.apk`. Debug APKs are for
testing and are signed with a development key. Store distribution requires your release
signing configuration. Build artifacts are attached to the GitHub release and CI runs.

## iPhone

The committed Swift Package Manager Xcode project is `ios/App/App.xcodeproj`.
On macOS with Xcode 16 or newer:

```bash
npm ci
npm run build
npx cap sync ios
npm run ios
```

Select the **App** scheme and an iPhone simulator. For a physical iPhone, select your
Apple development team in Signing & Capabilities and use a unique bundle identifier
if needed. The repository uses `com.josesanchez.agentfield`.

The macOS CI job builds an unsigned iOS simulator app. An App Store/TestFlight package
requires Apple signing and account access; a simulator build does not provide an installable IPA.

## Reproduce the recording

With the local app running:

```bash
PLAYWRIGHT_BROWSERS_PATH=/tmp/agent-field-playwright-cache npx playwright install ffmpeg
PLAYWRIGHT_BROWSERS_PATH=/tmp/agent-field-playwright-cache npm run demo
ffmpeg -y -i demo/feature-demo.webm -c:v libx264 -pix_fmt yuv420p -movflags +faststart demo/feature-demo.mp4
ffmpeg -y -i demo/feature-demo.mp4 -filter_complex "fps=8,scale=312:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" demo/feature-demo.gif
```

[Validation](demo/validation.json) records build/test scope.
The fixture provenance is in [`src/fixtures/meta.json`](src/fixtures/meta.json) and
[`src/fixtures/soccer.json`](src/fixtures/soccer.json). The tests check source bytes,
response contracts, probability validity, error states, and token persistence.
Fonts retain their SIL Open Font License; see [third-party notices](THIRD_PARTY_NOTICES.md).
