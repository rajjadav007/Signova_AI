import React from "react";
import { Cpu, Info } from "lucide-react";

export const MODEL_CONFIG = [
  { label: "Model", value: "Bi-GRU" },
  { label: "Architecture", value: "2-layer Bidirectional GRU" },
  { label: "Hidden Dimension", value: "128" },
  { label: "Input Features", value: "600" },
  { label: "Temporal Sequence Length", value: "64" },
  { label: "Supported Classes", value: "10" },
];

export default function ModelConfigCard() {
  return (
    <article className="dashboard-card" aria-labelledby="model-config-title">
      <div className="dashboard-card-header">
        <div className="dashboard-card-title-wrap">
          <div className="overview-card-icon" aria-hidden="true">
            <Cpu size={18} />
          </div>
          <h2 id="model-config-title" className="dashboard-card-title">
            Model Architecture
          </h2>
        </div>
        <span className="dashboard-card-badge">Configuration Spec</span>
      </div>

      <div className="model-specs-grid">
        {MODEL_CONFIG.map((spec) => (
          <div key={spec.label} className="model-spec-item">
            <span className="spec-label">{spec.label}</span>
            <span className="spec-value">{spec.value}</span>
          </div>
        ))}
      </div>

      <div className="model-notice">
        <div style={{ display: "flex", gap: "8px", alignItems: "flex-start" }}>
          <Info size={16} style={{ minWidth: "16px", marginTop: "2px", color: "var(--color-primary)" }} />
          <span>
            <strong>Configuration Information:</strong> Weights are loaded on demand by the backend inference service when processing sign recognition requests.
          </span>
        </div>
      </div>
    </article>
  );
}
