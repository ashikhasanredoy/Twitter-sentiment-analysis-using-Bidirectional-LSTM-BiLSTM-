document.addEventListener("DOMContentLoaded", () => {
  const apiStatusEl = document.getElementById("api-status");
  const statusLabel = document.getElementById("status-label");
  
  const tweetInput = document.getElementById("tweet-input");
  const charCounter = document.getElementById("char-counter");
  const btnPredict = document.getElementById("btn-predict");
  
  const resultPlaceholder = document.getElementById("result-placeholder");
  const resultContent = document.getElementById("result-content");
  const sentimentBanner = document.getElementById("sentiment-banner");
  const sentimentLabel = document.getElementById("sentiment-label");
  const confidenceVal = document.getElementById("confidence-val");
  
  const barPos = document.getElementById("bar-pos");
  const barNeu = document.getElementById("bar-neu");
  const barNeg = document.getElementById("bar-neg");
  const pctPos = document.getElementById("pct-pos");
  const pctNeu = document.getElementById("pct-neu");
  const pctNeg = document.getElementById("pct-neg");
  const cleanedTextDisplay = document.getElementById("cleaned-text-display");

  const batchInput = document.getElementById("batch-input");
  const btnBatchPredict = document.getElementById("btn-batch-predict");
  const batchResultsContainer = document.getElementById("batch-results-container");
  const batchTableBody = document.getElementById("batch-table-body");

  const tabBtns = document.querySelectorAll(".segment-btn");
  const tabContents = document.querySelectorAll(".tab-pane");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }
    });
  });

  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("API unreachable");
      const data = await res.json();

      apiStatusEl.classList.remove("ready", "warning");
      if (data.model_ready) {
        apiStatusEl.classList.add("ready");
        statusLabel.textContent = "BiLSTM Ready";
      } else {
        apiStatusEl.classList.add("warning");
        statusLabel.textContent = "Model Not Yet Trained";
      }
    } catch {
      apiStatusEl.classList.remove("ready");
      apiStatusEl.classList.add("warning");
      statusLabel.textContent = "API Offline";
    }
  }

  checkHealth();

  tweetInput.addEventListener("input", () => {
    charCounter.textContent = `${tweetInput.value.length} / 300`;
  });

  document.querySelectorAll(".pill-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const sample = btn.getAttribute("data-text");
      tweetInput.value = sample;
      charCounter.textContent = `${sample.length} / 300`;
      predictSingle();
    });
  });

  async function predictSingle() {
    const text = tweetInput.value.trim();
    if (!text) {
      alert("Please enter text to analyze.");
      return;
    }

    btnPredict.disabled = true;
    btnPredict.textContent = "Analyzing...";

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Prediction failed");
      }

      const data = await res.json();
      renderPrediction(data);
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      btnPredict.disabled = false;
      btnPredict.textContent = "Analyze Sentiment";
    }
  }

  function renderPrediction(data) {
    resultPlaceholder.classList.add("hidden");
    resultContent.classList.remove("hidden");

    cleanedTextDisplay.textContent = data.cleaned_text || "-";

    const sentiment = data.sentiment.toLowerCase();
    sentimentBanner.className = `score-banner ${sentiment}`;

    sentimentLabel.textContent = sentiment.toUpperCase();
    confidenceVal.textContent = `${Math.round(data.confidence * 100)}%`;

    const probs = data.probabilities || {};
    const pos = (probs.positive || 0) * 100;
    const neu = (probs.neutral || 0) * 100;
    const neg = (probs.negative || 0) * 100;

    barPos.style.width = `${pos}%`;
    pctPos.textContent = `${pos.toFixed(1)}%`;

    barNeu.style.width = `${neu}%`;
    pctNeu.textContent = `${neu.toFixed(1)}%`;

    barNeg.style.width = `${neg}%`;
    pctNeg.textContent = `${neg.toFixed(1)}%`;
  }

  btnPredict.addEventListener("click", predictSingle);

  btnBatchPredict.addEventListener("click", async () => {
    const raw = batchInput.value.trim();
    if (!raw) {
      alert("Please enter at least one text line.");
      return;
    }

    const lines = raw.split("\n").map(l => l.trim()).filter(Boolean);

    btnBatchPredict.disabled = true;
    btnBatchPredict.textContent = "Processing...";

    try {
      const res = await fetch("/api/predict/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: lines })
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Batch prediction failed");
      }

      const data = await res.json();
      batchTableBody.innerHTML = "";
      
      data.predictions.forEach(p => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${escapeHtml(p.text)}</td>
          <td><code>${escapeHtml(p.cleaned_text)}</code></td>
          <td><span class="count-badge">${escapeHtml(p.sentiment.toUpperCase())}</span></td>
          <td><strong>${Math.round(p.confidence * 100)}%</strong></td>
        `;
        batchTableBody.appendChild(tr);
      });

      batchResultsContainer.classList.remove("hidden");
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      btnBatchPredict.disabled = false;
      btnBatchPredict.textContent = "Run Batch";
    }
  });

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
});
