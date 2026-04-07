"""FastAPI application for llm-regression-detector."""

from fastapi import Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from models import LlmRewardLabAction, LlmRewardLabObservation
from openenv.core.env_server.http_server import create_app
from server.environment import LlmRewardLabEnvironment
from server.graders import grade_submission
from server.tasks import TASK_REGISTRY, list_tasks

app = create_app(
    LlmRewardLabEnvironment,
    LlmRewardLabAction,
    LlmRewardLabObservation,
    env_name="llm-regression-detector",
    max_concurrent_envs=4,
)

# Remove OpenEnv's default routes (including Gradio UI at /web) so only our
# custom dashboard is served.
_remove_paths = {"/", "/reset", "/step", "/state", "/web", "/web/"}
app.routes[:] = [
    r for r in app.routes
    if getattr(r, "path", None) not in _remove_paths
    and not getattr(r, "path", "").startswith("/web")
]


class GraderRequest(BaseModel):
    task_id: str
    drift_events: list[str] = Field(default_factory=list)
    remediations: list[str] = Field(default_factory=list)
    budget_used: int = 0
    budget_total: int | None = None


@app.get("/tasks")
def tasks_endpoint():
    return {
        "tasks": list_tasks(),
        "action_schema": LlmRewardLabAction.model_json_schema(),
    }


@app.post("/grader")
def grader_endpoint(payload: GraderRequest = Body(...)):
    if payload.task_id not in TASK_REGISTRY:
        return {"error": f"Unknown task_id: {payload.task_id}"}

    cfg = TASK_REGISTRY[payload.task_id]
    score = grade_submission(
        task_id=payload.task_id,
        submitted_drifts=payload.drift_events,
        submitted_remediations=payload.remediations,
        active_drifts=cfg["active_drifts"],
        budget_used=payload.budget_used,
        budget_total=(
            cfg["budget"] if payload.budget_total is None else payload.budget_total
        ),
    )
    return {"task_id": payload.task_id, "score": score}


@app.post("/baseline")
def baseline_endpoint():
    from inference import run_baseline

    return run_baseline()


# ---------------------------------------------------------------------------
# Stateful dashboard API (OpenEnv HTTP is stateless per-request)
# ---------------------------------------------------------------------------

_dashboard_env = LlmRewardLabEnvironment()


class DashboardResetRequest(BaseModel):
    task_id: str = "task_detect_localize"
    seed: int = 42


@app.post("/reset")
def dashboard_reset(req: DashboardResetRequest = Body(...)):
    obs = _dashboard_env.reset(task_id=req.task_id, seed=req.seed)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


class DashboardStepRequest(BaseModel):
    action_type: str
    parameters: dict = Field(default_factory=dict)


@app.post("/step")
def dashboard_step(req: DashboardStepRequest = Body(...)):
    action = LlmRewardLabAction(action_type=req.action_type, parameters=req.parameters)
    obs = _dashboard_env.step(action)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.get("/state")
def dashboard_state():
    return _dashboard_env.state.model_dump() if _dashboard_env.state else {}


# ---------------------------------------------------------------------------
# Embedded dashboard UI
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Regression Detector</title>
<style>
  :root {
    --bg: #141517;
    --surface: #1e1f23;
    --surface2: #25262b;
    --border: #2c2e33;
    --text: #e9ecef;
    --text-dim: #868e96;
    --accent: #4c6ef5;
    --accent-hover: #5c7cfa;
    --green: #40c057;
    --red: #fa5252;
    --orange: #fd7e14;
    --yellow: #fab005;
    --radius: 8px;
    --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
    --mono: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    min-height: 100vh;
    padding: 1rem;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* Layout */
  .container { max-width: 1200px; margin: 0 auto; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; }
  @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }
  @media (max-width: 700px) { .grid-4 { grid-template-columns: repeat(2, 1fr); } }

  /* Header */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.25rem; flex-wrap: wrap; gap: 0.5rem;
  }
  .header h1 { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; }
  .header .subtitle { color: var(--text-dim); font-size: 0.8rem; }
  .header-links { display: flex; gap: 1rem; font-size: 0.8rem; }

  /* Cards */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
  }
  .card-title {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-dim); margin-bottom: 0.5rem;
  }

  /* Metric cards */
  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
    text-align: center;
  }
  .metric-card .label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); }
  .metric-card .value { font-size: 1.5rem; font-weight: 700; font-family: var(--mono); margin-top: 0.15rem; }
  .metric-card .value.blue { color: var(--accent); }
  .metric-card .value.orange { color: var(--orange); }
  .metric-card .value.green { color: var(--green); }
  .metric-card .value.red { color: var(--red); }

  /* Status pill */
  .status-pill {
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em;
  }
  .status-pill.running { background: rgba(64,192,87,0.15); color: var(--green); }
  .status-pill.done { background: rgba(134,142,150,0.15); color: var(--text-dim); }
  .status-pill.idle { background: rgba(76,110,245,0.15); color: var(--accent); }

  /* Pills row */
  .pills { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.75rem; }
  .pill {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.2rem 0.5rem; font-size: 0.72rem;
    display: flex; gap: 0.3rem; align-items: center;
  }
  .pill kbd { color: var(--text-dim); font-family: var(--font); font-size: 0.68rem; }
  .pill .val { color: var(--text); font-weight: 600; }

  /* Incident message */
  .incident-msg {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 0.75rem 1rem;
    font-size: 0.85rem; line-height: 1.6; white-space: pre-wrap;
    max-height: 6rem; overflow-y: auto;
  }

  /* Form elements */
  label { font-size: 0.72rem; color: var(--text-dim); display: block; margin-bottom: 0.2rem; }
  select, input[type="number"], input[type="text"], textarea {
    width: 100%; background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.45rem 0.6rem; font-size: 0.82rem; font-family: var(--font);
    outline: none; transition: border-color 0.15s;
  }
  select:focus, input:focus, textarea:focus { border-color: var(--accent); }
  textarea { resize: vertical; min-height: 3rem; font-family: var(--mono); font-size: 0.78rem; }

  .field-group { display: none; }
  .field-group.active { display: block; }
  .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.5rem; }
  .form-row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.5rem; margin-bottom: 0.5rem; }

  /* JSON preview */
  .json-preview {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.6rem; font-family: var(--mono);
    font-size: 0.75rem; max-height: 8rem; overflow-y: auto;
    white-space: pre-wrap; word-break: break-all; color: var(--text-dim);
    margin-top: 0.5rem;
  }

  /* Buttons */
  .btn {
    padding: 0.5rem 1rem; border: none; border-radius: 4px;
    font-size: 0.8rem; font-weight: 600; cursor: pointer; transition: all 0.15s;
    font-family: var(--font);
  }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover:not(:disabled) { background: var(--accent-hover); }
  .btn-execute { background: #e8590c; color: #fff; }
  .btn-execute:hover:not(:disabled) { background: #d9480f; }
  .btn-reset { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
  .btn-reset:hover { background: var(--border); }
  .btn-group { display: flex; gap: 0.5rem; margin-top: 0.75rem; flex-wrap: wrap; }

  /* Timeline */
  .timeline { max-height: 16rem; overflow-y: auto; }
  .timeline-empty { color: var(--text-dim); font-size: 0.8rem; padding: 1rem 0; text-align: center; }
  .timeline-item {
    padding: 0.5rem 0; border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
  }
  .timeline-item:last-child { border-bottom: none; }
  .timeline-row { display: flex; justify-content: space-between; align-items: center; }
  .tl-step-num { color: var(--text-dim); }
  .tl-action-name { font-weight: 600; }
  .tl-reward { font-family: var(--mono); font-weight: 700; font-size: 0.82rem; }
  .tl-reward.pos { color: var(--green); }
  .tl-reward.neg { color: var(--red); }
  .tl-feedback { color: var(--text-dim); font-size: 0.75rem; margin-top: 0.2rem; }

  /* Output panels */
  .output-panel {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.6rem; font-family: var(--mono);
    font-size: 0.72rem; max-height: 22rem; overflow-y: auto;
    white-space: pre-wrap; word-break: break-all; line-height: 1.5;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* Toast */
  .toast-container { position: fixed; top: 1rem; right: 1rem; z-index: 9999; display: flex; flex-direction: column; gap: 0.5rem; }
  .toast {
    background: var(--red); color: #fff; padding: 0.6rem 1rem;
    border-radius: var(--radius); font-size: 0.8rem; max-width: 360px;
    transform: translateX(110%); transition: transform 0.28s ease;
    cursor: pointer;
  }
  .toast.visible { transform: translateX(0); }

  .section-gap { margin-top: 1rem; }
  .mb-05 { margin-bottom: 0.5rem; }
</style>
</head>
<body>
<div id="toasts" class="toast-container"></div>

<div class="container">

<!-- HEADER -->
<div class="header">
  <div>
    <h1>LLM Regression Detector</h1>
    <div class="subtitle">Inspect &rarr; investigate &rarr; diagnose &rarr; remediate &middot; <a href="/health">/health</a></div>
  </div>
  <div class="header-links">
    <a href="/docs">Docs</a>
    <a href="/tasks">Tasks</a>
  </div>
</div>

<!-- METRICS -->
<div class="grid-4 mb-05">
  <div class="metric-card">
    <div class="label">Last Step Reward</div>
    <div class="value blue" id="metricLastReward">--</div>
  </div>
  <div class="metric-card">
    <div class="label">Total Reward</div>
    <div class="value orange" id="metricTotalReward">--</div>
  </div>
  <div class="metric-card">
    <div class="label">Score</div>
    <div class="value green" id="metricScore">--</div>
  </div>
  <div class="metric-card">
    <div class="label">Status</div>
    <div id="metricStatus"><span class="status-pill idle">IDLE</span></div>
  </div>
</div>

<!-- MAIN GRID -->
<div class="grid-2 section-gap">

  <!-- LEFT COLUMN -->
  <div>
    <!-- Current Task -->
    <div class="card mb-05">
      <div class="card-title">Current Task</div>
      <div class="pills" id="taskPills">
        <div class="pill"><kbd>Status:</kbd><span class="val">No episode</span></div>
      </div>
      <div class="incident-msg" id="incidentMsg">Click Reset to start an episode.</div>
    </div>

    <!-- Business Metrics -->
    <div class="card mb-05">
      <div class="card-title">Business Metrics</div>
      <div class="pills" id="bizPills">
        <div class="pill"><kbd>Satisfaction:</kbd><span class="val">--</span></div>
        <div class="pill"><kbd>Error Rate:</kbd><span class="val">--</span></div>
        <div class="pill"><kbd>Cost/1k:</kbd><span class="val">--</span></div>
      </div>
    </div>

    <!-- Action Timeline -->
    <div class="card">
      <div class="card-title">Action Timeline</div>
      <div class="timeline" id="timeline">
        <div class="timeline-empty">No actions yet -- reset to begin.</div>
      </div>
    </div>
  </div>

  <!-- RIGHT COLUMN -->
  <div>
    <!-- Action Builder -->
    <div class="card mb-05">
      <div class="card-title">Step-by-step Action</div>

      <div class="form-row mb-05">
        <div>
          <label>Action type</label>
          <select id="actionType" onchange="showFieldGroups()">
            <option value="inspect_samples">inspect_samples</option>
            <option value="run_ab_test">run_ab_test</option>
            <option value="run_targeted_eval">run_targeted_eval</option>
            <option value="submit_diagnosis">submit_diagnosis</option>
          </select>
        </div>
        <div></div>
      </div>

      <!-- inspect_samples fields -->
      <div class="field-group active" id="fgInspect_samples">
        <div class="form-row-3">
          <div>
            <label>task_type</label>
            <select id="inspTaskType" onchange="syncPreview()">
              <option value="">(all)</option>
              <option value="summarization">summarization</option>
              <option value="qa">qa</option>
              <option value="coding">coding</option>
              <option value="translation">translation</option>
              <option value="classification">classification</option>
            </select>
          </div>
          <div>
            <label>input_length</label>
            <select id="inspInputLen" onchange="syncPreview()">
              <option value="">(all)</option>
              <option value="short">short</option>
              <option value="medium">medium</option>
              <option value="long">long</option>
            </select>
          </div>
          <div>
            <label>limit</label>
            <input type="number" id="inspLimit" value="20" min="1" max="50" oninput="syncPreview()">
          </div>
        </div>
      </div>

      <!-- run_ab_test fields -->
      <div class="field-group" id="fgRun_ab_test">
        <div class="form-row">
          <div>
            <label>hypothesis_id</label>
            <select id="abHypothesis" onchange="syncPreview()">
              <option value="">-- select --</option>
            </select>
          </div>
          <div></div>
        </div>
      </div>

      <!-- run_targeted_eval fields -->
      <div class="field-group" id="fgRun_targeted_eval">
        <div class="form-row-3">
          <div>
            <label>task_type</label>
            <select id="evalTaskType" onchange="syncPreview()">
              <option value="">(all)</option>
              <option value="summarization">summarization</option>
              <option value="qa">qa</option>
              <option value="coding">coding</option>
              <option value="translation">translation</option>
              <option value="classification">classification</option>
            </select>
          </div>
          <div>
            <label>input_length</label>
            <select id="evalInputLen" onchange="syncPreview()">
              <option value="">(all)</option>
              <option value="short">short</option>
              <option value="medium">medium</option>
              <option value="long">long</option>
            </select>
          </div>
          <div>
            <label>count</label>
            <input type="number" id="evalCount" value="6" min="1" max="20" oninput="syncPreview()">
          </div>
        </div>
      </div>

      <!-- submit_diagnosis fields -->
      <div class="field-group" id="fgSubmit_diagnosis">
        <div class="mb-05">
          <label>drift_events (comma-separated)</label>
          <input type="text" id="submitDrifts" placeholder="e.g. data_contamination, quantization_applied" oninput="syncPreview()">
        </div>
        <div class="mb-05">
          <label>remediations (comma-separated)</label>
          <input type="text" id="submitRemediations" placeholder="e.g. rollback_finetune_checkpoint, revert_quantization" oninput="syncPreview()">
        </div>
        <div>
          <label>explanation (optional, earns bonus)</label>
          <textarea id="submitExplanation" rows="2" placeholder="Why you believe these drifts are active..." oninput="syncPreview()"></textarea>
        </div>
      </div>

      <div class="json-preview" id="jsonPreview">{}</div>

      <!-- Controls -->
      <div style="margin-top:0.75rem; border-top:1px solid var(--border); padding-top:0.75rem;">
        <div class="form-row-3 mb-05">
          <div>
            <label>Task</label>
            <select id="resetTask">
              <option value="task_detect_localize">easy (detect_localize)</option>
              <option value="task_diagnose">medium (diagnose)</option>
              <option value="task_multi_drift">hard (multi_drift)</option>
            </select>
          </div>
          <div>
            <label>Seed</label>
            <input type="number" id="resetSeed" value="42" min="0">
          </div>
          <div></div>
        </div>
        <div class="btn-group">
          <button class="btn btn-execute" id="btnStep" onclick="doStep()" disabled>Execute Step</button>
          <button class="btn btn-reset" onclick="doReset()">Reset</button>
          <button class="btn btn-primary" onclick="doGetState()">Get State</button>
          <button class="btn btn-reset" onclick="doBaseline()">Run Baseline</button>
        </div>
      </div>
    </div>

    <!-- Output -->
    <div class="card">
      <div class="card-title">Response</div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.4rem; margin-bottom:0.4rem;">
        <div>
          <label>reward</label>
          <div class="output-panel" id="outReward" style="max-height:2rem;">--</div>
        </div>
        <div>
          <label>done</label>
          <div class="output-panel" id="outDone" style="max-height:2rem;">--</div>
        </div>
      </div>
      <label>observation</label>
      <div class="output-panel" id="outObs">--</div>
    </div>
  </div>
</div>

</div>

<script>
// ── State ──
let totalReward = 0;
let episodeActive = false;
let stepHistory = [];

function syncStepBtn() {
  document.getElementById("btnStep").disabled = !episodeActive;
}

// ── Toast ──
function showToast(msg) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  t.onclick = () => t.remove();
  document.getElementById("toasts").appendChild(t);
  requestAnimationFrame(() => t.classList.add("visible"));
  setTimeout(() => { t.classList.remove("visible"); setTimeout(() => t.remove(), 300); }, 6000);
}

// ── Field groups ──
function showFieldGroups() {
  const at = document.getElementById("actionType").value;
  document.querySelectorAll(".field-group").forEach(el => {
    const key = el.id.replace("fg", "");
    el.classList.toggle("active", key.toLowerCase() === at.toLowerCase()
      || (at === "inspect_samples" && key === "Inspect_samples")
      || (at === "run_ab_test" && key === "Run_ab_test")
      || (at === "run_targeted_eval" && key === "Run_targeted_eval")
      || (at === "submit_diagnosis" && key === "Submit_diagnosis")
    );
  });
  syncPreview();
}

// ── Action builder ──
function buildAction() {
  const at = document.getElementById("actionType").value;
  if (at === "inspect_samples") {
    const p = {};
    const tt = document.getElementById("inspTaskType").value;
    const il = document.getElementById("inspInputLen").value;
    const lim = parseInt(document.getElementById("inspLimit").value, 10);
    if (tt) p.task_type = tt;
    if (il) p.input_length = il;
    p.limit = lim || 20;
    return { action_type: "inspect_samples", parameters: p };
  }
  if (at === "run_ab_test") {
    return { action_type: "run_ab_test", parameters: { hypothesis_id: document.getElementById("abHypothesis").value } };
  }
  if (at === "run_targeted_eval") {
    const p = {};
    const tt = document.getElementById("evalTaskType").value;
    const il = document.getElementById("evalInputLen").value;
    const c = parseInt(document.getElementById("evalCount").value, 10);
    if (tt) p.task_type = tt;
    if (il) p.input_length = il;
    p.count = c || 6;
    return { action_type: "run_targeted_eval", parameters: p };
  }
  if (at === "submit_diagnosis") {
    const drifts = document.getElementById("submitDrifts").value.split(",").map(s => s.trim()).filter(Boolean);
    const rems = document.getElementById("submitRemediations").value.split(",").map(s => s.trim()).filter(Boolean);
    const expl = document.getElementById("submitExplanation").value.trim();
    const p = { drift_events: drifts, remediations: rems };
    if (expl) p.explanation = expl;
    return { action_type: "submit_diagnosis", parameters: p };
  }
  return { action_type: at, parameters: {} };
}

function syncPreview() {
  document.getElementById("jsonPreview").textContent = JSON.stringify(buildAction(), null, 2);
}

// ── Render functions ──
function renderMetrics(reward, done) {
  document.getElementById("metricLastReward").textContent = reward.toFixed(4);
  document.getElementById("metricTotalReward").textContent = totalReward.toFixed(4);
  const score = Math.min(totalReward, 1.0);
  document.getElementById("metricScore").textContent = (score * 100).toFixed(0) + "%";
  const statusEl = document.getElementById("metricStatus");
  if (!episodeActive && totalReward === 0) {
    statusEl.innerHTML = '<span class="status-pill idle">IDLE</span>';
  } else if (done) {
    statusEl.innerHTML = '<span class="status-pill done">DONE</span>';
  } else {
    statusEl.innerHTML = '<span class="status-pill running">RUNNING</span>';
  }
}

function renderTask(obs) {
  const pills = document.getElementById("taskPills");
  const msg = document.getElementById("incidentMsg");
  pills.innerHTML = "";

  const addPill = (k, v) => {
    if (v === undefined || v === null) return;
    const d = document.createElement("div");
    d.className = "pill";
    d.innerHTML = '<kbd>' + k + ':</kbd><span class="val">' + v + '</span>';
    pills.appendChild(d);
  };

  addPill("Task", obs.task_id || "--");
  addPill("Step", obs.step_count + "/" + (obs.metadata?.max_steps || "?"));
  addPill("Budget", obs.budget_remaining + "/" + obs.budget_total);

  if (obs.incident_context) {
    const ic = obs.incident_context;
    addPill("Severity", ic.severity);
    addPill("Affected", ic.affected_users_percent + "%");
    addPill("Minutes ago", ic.started_minutes_ago);
    msg.textContent = ic.reported_issue || "No incident details.";
  } else {
    msg.textContent = obs.last_action_result || "No incident details.";
  }

  // Populate hypothesis dropdown
  if (obs.available_hypotheses && obs.available_hypotheses.length) {
    const sel = document.getElementById("abHypothesis");
    const cur = sel.value;
    sel.innerHTML = '<option value="">-- select --</option>';
    obs.available_hypotheses.forEach(h => {
      const o = document.createElement("option");
      o.value = h.hypothesis_id;
      o.textContent = h.hypothesis_id + " (cost: " + h.cost + ")";
      sel.appendChild(o);
    });
    if (cur) sel.value = cur;
  }
}

function renderBizMetrics(obs) {
  const bm = obs.business_metrics;
  const el = document.getElementById("bizPills");
  if (!bm) { el.innerHTML = '<div class="pill"><kbd>No data</kbd></div>'; return; }
  el.innerHTML = "";
  const addPill = (k, v) => {
    const d = document.createElement("div");
    d.className = "pill";
    d.innerHTML = '<kbd>' + k + ':</kbd><span class="val">' + v + '</span>';
    el.appendChild(d);
  };
  addPill("Satisfaction", bm.user_satisfaction);
  addPill("Error Rate", bm.error_rate);
  addPill("Cost/1k", "$" + bm.cost_per_1k_requests);
}

function renderTimeline() {
  const el = document.getElementById("timeline");
  if (!stepHistory.length) {
    el.innerHTML = '<div class="timeline-empty">No actions yet -- reset to begin.</div>';
    return;
  }
  el.innerHTML = "";
  stepHistory.forEach((s, i) => {
    const item = document.createElement("div");
    item.className = "timeline-item";
    const cls = s.reward >= 0 ? "pos" : "neg";
    const sign = s.reward >= 0 ? "+" : "";
    item.innerHTML =
      '<div class="timeline-row">' +
        '<div class="tl-main"><span class="tl-step-num">Step ' + i + ': </span><span class="tl-action-name">' + s.action + '</span></div>' +
        '<div class="tl-reward ' + cls + '">' + sign + s.reward.toFixed(4) + '</div>' +
      '</div>' +
      (s.feedback ? '<div class="tl-feedback">' + s.feedback + '</div>' : '');
    el.appendChild(item);
  });
  el.scrollTop = el.scrollHeight;
}

function renderOutput(data) {
  document.getElementById("outReward").textContent = data.reward !== undefined ? data.reward : "--";
  document.getElementById("outDone").textContent = data.done !== undefined ? String(data.done) : "--";
  document.getElementById("outObs").textContent = JSON.stringify(data, null, 2);
}

// ── API calls ──
// OpenEnv wraps responses as {observation: {...}, reward, done}
// Flatten so the rest of our code can access fields directly
function unwrap(data) {
  if (data.observation) {
    const obs = data.observation;
    obs.reward = data.reward;
    obs.done = data.done;
    return obs;
  }
  return data;
}

async function doReset() {
  const task = document.getElementById("resetTask").value;
  const seed = parseInt(document.getElementById("resetSeed").value, 10);
  try {
    const res = await fetch("/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: task, seed: seed }),
    });
    if (!res.ok) throw new Error("Reset failed: " + res.status);
    const raw = await res.json();
    const data = unwrap(raw);
    totalReward = 0;
    stepHistory = [];
    episodeActive = !data.done;
    syncStepBtn();
    renderMetrics(0, false);
    renderTask(data);
    renderBizMetrics(data);
    renderTimeline();
    renderOutput(raw);
  } catch (e) {
    showToast(e.message);
  }
}

async function doStep() {
  const action = buildAction();
  try {
    const res = await fetch("/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(action),
    });
    if (!res.ok) throw new Error("Step failed: " + res.status);
    const raw = await res.json();
    const data = unwrap(raw);
    const reward = data.reward || 0;
    totalReward += reward;
    if (data.done) episodeActive = false;
    stepHistory.push({
      action: action.action_type,
      reward: reward,
      feedback: data.last_action_result || "",
    });
    syncStepBtn();
    renderMetrics(reward, data.done);
    renderTask(data);
    renderBizMetrics(data);
    renderTimeline();
    renderOutput(raw);
  } catch (e) {
    showToast(e.message);
  }
}

async function doGetState() {
  try {
    const res = await fetch("/state");
    if (!res.ok) throw new Error("State failed: " + res.status);
    const data = await res.json();
    renderOutput(data);
  } catch (e) {
    showToast(e.message);
  }
}

async function doBaseline() {
  try {
    const res = await fetch("/baseline", { method: "POST" });
    if (!res.ok) throw new Error("Baseline failed: " + res.status);
    const data = await res.json();
    renderOutput(data);
  } catch (e) {
    showToast(e.message);
  }
}

// Init
showFieldGroups();
syncPreview();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard_ui():
    return _DASHBOARD_HTML

@app.get("/web/", response_class=HTMLResponse)
async def dashboard_ui_web():
    return _DASHBOARD_HTML

@app.get("/web", response_class=HTMLResponse)
async def dashboard_ui_web_2():
    return _DASHBOARD_HTML

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()