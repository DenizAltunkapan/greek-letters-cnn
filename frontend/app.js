const greekLetters = [
  { name: "Alpha", latex: "\\alpha", fallback: "α" },
  { name: "Beta", latex: "\\beta", fallback: "β" },
  { name: "Gamma", latex: "\\gamma", fallback: "γ" },
  { name: "Delta", latex: "\\delta", fallback: "δ" },
  { name: "Epsilon", latex: "\\epsilon", fallback: "ε" },
  { name: "Zeta", latex: "\\zeta", fallback: "ζ" },
  { name: "Eta", latex: "\\eta", fallback: "η" },
  { name: "Theta", latex: "\\theta", fallback: "θ" },
  { name: "Iota", latex: "\\iota", fallback: "ι" },
  { name: "Kappa", latex: "\\kappa", fallback: "κ" },
  { name: "Lambda", latex: "\\lambda", fallback: "λ" },
  { name: "Mu", latex: "\\mu", fallback: "μ" },
  { name: "Nu", latex: "\\nu", fallback: "ν" },
  { name: "Xi", latex: "\\xi", fallback: "ξ" },
  { name: "Omicron", latex: "o", fallback: "ο" },
  { name: "Pi", latex: "\\pi", fallback: "π" },
  { name: "Rho", latex: "\\rho", fallback: "ρ" },
  { name: "Sigma", latex: "\\sigma", fallback: "σ" },
  { name: "Tau", latex: "\\tau", fallback: "τ" },
  { name: "Upsilon", latex: "\\upsilon", fallback: "υ" },
  { name: "Phi", latex: "\\varphi", fallback: "ϕ" },
  { name: "Chi", latex: "\\chi", fallback: "χ" },
  { name: "Psi", latex: "\\psi", fallback: "ψ" },
  { name: "Omega", latex: "\\omega", fallback: "ω" },
];

const greekTableBody = document.querySelector("#smallGreekTable tbody");
const modelInfoEl = document.getElementById("modelInfo");
const archPlotEl = document.getElementById("archPlot");
const refreshPlotBtn = document.getElementById("refreshPlot");
const predictionResultEl = document.getElementById("predictionResult");
const topPredictionsEl = document.getElementById("topPredictions");

const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");

function renderGreekLetters() {
  greekTableBody.innerHTML = "";
  greekLetters.forEach((item) => {
    const row = document.createElement("tr");

    const symbolCell = document.createElement("td");
    symbolCell.className = "symbol";
    const symbolContent = document.createElement("span");
    symbolContent.className = "symbol-glyph";

    if (window.katex && item.latex) {
      symbolContent.innerHTML = katex.renderToString(item.latex, {
        throwOnError: false,
      });
    } else {
      symbolContent.textContent = item.fallback || "";
    }
    symbolCell.appendChild(symbolContent);

    const nameCell = document.createElement("td");
    nameCell.textContent = item.name;

    row.appendChild(symbolCell);
    row.appendChild(nameCell);
    greekTableBody.appendChild(row);
  });
}

async function fetchModelInfo() {
  modelInfoEl.textContent = "Loading model metadata...";
  try {
    const response = await fetch("/api/model-info");
    if (!response.ok) throw new Error("Could not load model info.");
    const data = await response.json();

    const cfg = data.architecture || {};
    modelInfoEl.innerHTML = `
      <p><strong>Selection Metric:</strong> ${data.selection_metric || "n/a"}</p>
      <p><strong>Selection Score:</strong> ${typeof data.selection_score === "number" ? data.selection_score.toFixed(4) : "n/a"}</p>
      <p><strong>Seed:</strong> ${data.selected_seed ?? "n/a"}</p>
      <p><strong>Channels:</strong> ${Array.isArray(cfg.channels) ? cfg.channels.join(" → ") : "n/a"}</p>
      <p><strong>Hidden Size:</strong> ${cfg.hidden_size ?? "n/a"} | <strong>Dropout:</strong> ${cfg.dropout ?? "n/a"}</p>
      <p><strong>Classes:</strong> ${data.num_classes ?? "n/a"}</p>
    `;
  } catch (error) {
    modelInfoEl.textContent = `Failed to load model metadata: ${error.message}`;
  }
}

function fetchArchitecturePlot() {
  archPlotEl.src = `/api/architecture-comparison?t=${Date.now()}`;
}

function initCanvas() {
  const width = canvas.width;
  const height = canvas.height;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.lineWidth = 13;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#111827";

  let drawing = false;

  function getPos(event) {
    const rect = canvas.getBoundingClientRect();
    const clientX = event.touches ? event.touches[0].clientX : event.clientX;
    const clientY = event.touches ? event.touches[0].clientY : event.clientY;

    return {
      x: ((clientX - rect.left) * width) / rect.width,
      y: ((clientY - rect.top) * height) / rect.height,
    };
  }

  function startDraw(event) {
    drawing = true;
    const { x, y } = getPos(event);
    ctx.beginPath();
    ctx.moveTo(x, y);
    event.preventDefault();
  }

  function draw(event) {
    if (!drawing) return;
    const { x, y } = getPos(event);
    ctx.lineTo(x, y);
    ctx.stroke();
    event.preventDefault();
  }

  function endDraw(event) {
    if (!drawing) return;
    drawing = false;
    ctx.closePath();
    event.preventDefault();
  }

  canvas.addEventListener("mousedown", startDraw);
  canvas.addEventListener("mousemove", draw);
  window.addEventListener("mouseup", endDraw);

  canvas.addEventListener("touchstart", startDraw, { passive: false });
  canvas.addEventListener("touchmove", draw, { passive: false });
  window.addEventListener("touchend", endDraw, { passive: false });
}

function clearCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  predictionResultEl.textContent = "No prediction yet.";
  predictionResultEl.className = "result muted";
  topPredictionsEl.innerHTML = "";
}

function canvasToBlob() {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/png");
  });
}

async function runPrediction() {
  predictionResultEl.textContent = "Predicting...";
  predictionResultEl.className = "result muted";

  const blob = await canvasToBlob();
  const formData = new FormData();
  formData.append("file", blob, "drawing.png");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const details = await response.json().catch(() => ({}));
      throw new Error(details.detail || "Prediction failed.");
    }

    const data = await response.json();
    predictionResultEl.textContent = `Prediction: ${data.prediction} (${data.confidence.toFixed(2)}%)`;
    predictionResultEl.className = "result";

    topPredictionsEl.innerHTML = "";
    (data.top_predictions || []).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = `${item.label}: ${item.confidence.toFixed(2)}%`;
      topPredictionsEl.appendChild(li);
    });
  } catch (error) {
    predictionResultEl.textContent = `Error: ${error.message}`;
    predictionResultEl.className = "result muted";
  }
}

function wireEvents() {
  document.getElementById("clearCanvas").addEventListener("click", clearCanvas);
  document.getElementById("predictBtn").addEventListener("click", runPrediction);
  refreshPlotBtn.addEventListener("click", fetchArchitecturePlot);
}

renderGreekLetters();
initCanvas();
wireEvents();
fetchModelInfo();
fetchArchitecturePlot();
clearCanvas();
