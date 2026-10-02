document.addEventListener("DOMContentLoaded", () => {
  const apiStatusEl = document.getElementById("api-status");
  const statusLabel = document.getElementById("status-label");
  const modelSelect = document.getElementById("model-select");
  const batchModelSelect = document.getElementById("batch-model-select");
  
  const tweetInput = document.getElementById("tweet-input");
  const charCounter = document.getElementById("char-counter");
  const btnPredict = document.getElementById("btn-predict");
  
  const resultPlaceholder = document.getElementById("result-placeholder");
  const resultContent = document.getElementById("result-content");
  const modelTag = document.getElementById("model-tag");
  const sentimentBanner = document.getElementById("sentiment-banner");
  const sentimentIcon = document.getElementById("sentiment-icon");
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

  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

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
      if (!res.ok) throw new Error("API not reachable");
      const data = await res.json();
      
      const lstm = data.models?.lstm_ready;
      const bilstm = data.models?.bilstm_ready;

      apiStatusEl.classList.remove("ready", "warning");
      if (lstm && bilstm) {
        apiStatusEl.classList.add("ready");
        statusLabel.textContent = "LSTM & BiLSTM Ready";
      } else if (lstm || bilstm) {
        apiStatusEl.classList.add("ready");
        statusLabel.textContent = lstm ? "LSTM Ready" : "BiLSTM Ready";
      } else {
        apiStatusEl.classList.add("warning");
        statusLabel.textContent = "Models Not Yet Trained";
      }
    } catch (err) {
      apiStatusEl.classList.remove("ready");
      apiStatusEl.classList.add("warning");
      statusLabel.textContent = "Backend Offline";
    }
  }

  checkHealth();

  tweetInput.addEventListener("input", () => {
    const len = tweetInput.value.length;
    charCounter.textContent = `${len} / 300`;
  });

  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const sample = chip.getAttribute("data-text");
      tweetInput.value = sample;
      charCounter.textContent = `${sample.length} / 300`;
      predictSingle();
    });
  });

  async function predictSingle() {
    const text = tweetInput.value.trim();
    if (!text) {
      alert("Please enter a tweet text to analyze.");
      return;
    }

    const modelType = modelSelect.value;
    btnPredict.disabled = true;
    btnPredict.textContent = "Analyzing...";

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model_type: modelType })
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Prediction failed");
      }

      const data = await res.json();
      renderPrediction(data);
    } catch (err) {
      alert(`Prediction Error: ${err.message}`);
    } finally {
      btnPredict.disabled = false;
      btnPredict.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        Analyze Sentiment
      `;
    }
  }

  function renderPrediction(data) {
    resultPlaceholder.classList.add("hidden");
    resultContent.classList.remove("hidden");

    modelTag.textContent = data.model_type.toUpperCase();
    cleanedTextDisplay.textContent = data.cleaned_text || "(empty after cleaning)";

    const sentiment = data.sentiment.toLowerCase();
    sentimentBanner.className = `sentiment-banner ${sentiment}`;

    const icons = {
      positive: "😊",
      neutral: "😐",
      negative: "😔"
    };

    sentimentIcon.textContent = icons[sentiment] || "🔍";
    sentimentLabel.textContent = sentiment.toUpperCase();
    confidenceVal.textContent = `${(data.confidence * 100).toFixed(1)}%`;

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
      alert("Please enter at least one tweet line.");
      return;
    }

    const lines = raw.split("\n").map(l => l.trim()).filter(Boolean);
    const modelType = batchModelSelect.value;

    btnBatchPredict.disabled = true;
    btnBatchPredict.textContent = "Processing batch...";

    try {
      const res = await fetch("/api/predict/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: lines, model_type: modelType })
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Batch prediction failed");
      }

      const data = await res.json();
      batchTableBody.innerHTML = "";
      
      data.predictions.forEach(p => {
        const tr = document.createElement("tr");
        const sentLower = p.sentiment.toLowerCase();
        tr.innerHTML = `
          <td>${escapeHtml(p.text)}</td>
          <td><code>${escapeHtml(p.cleaned_text)}</code></td>
          <td><span class="badge" style="background: ${getBadgeColor(sentLower)}; color: #fff;">${p.sentiment}</span></td>
          <td><strong>${(p.confidence * 100).toFixed(1)}%</strong></td>
        `;
        batchTableBody.appendChild(tr);
      });

      batchResultsContainer.classList.remove("hidden");
    } catch (err) {
      alert(`Batch Error: ${err.message}`);
    } finally {
      btnBatchPredict.disabled = false;
      btnBatchPredict.textContent = "Run Batch Prediction";
    }
  });

  function getBadgeColor(sent) {
    if (sent === "positive") return "#10b981";
    if (sent === "negative") return "#f43f5e";
    return "#6366f1";
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
});
