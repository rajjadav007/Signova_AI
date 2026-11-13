import React from "react";
import { Layers, CheckSquare, Timer, ShieldAlert } from "lucide-react";

const RECOGNITION_ITEMS = [
  {
    title: "Primary Recognition Engine",
    desc: "Individual word recognition using bidirectional gated recurrent units (Bi-GRU) optimized for temporal gesture transitions.",
    icon: Layers,
  },
  {
    title: "10 Supported Classes",
    desc: "Targeted vocabulary covering essential everyday ISL communication: greetings, responses, needs, and emergency signs.",
    icon: CheckSquare,
  },
  {
    title: "Capture-and-Recognize",
    desc: "3-second structured recording window with real-time countdown, camera preview, and automated landmark extraction.",
    icon: Timer,
  },
  {
    title: "Confidence Rejection",
    desc: "Dual-check threshold (0.80 score) tags ambiguous or low-volume hand signs as uncertain rather than misclassifying.",
    icon: ShieldAlert,
  },
];

export default function RecognitionOverview() {
  return (
    <article className="dashboard-card" aria-labelledby="recognition-overview-title">
      <div className="dashboard-card-header">
        <h2 id="recognition-overview-title" className="dashboard-card-title">
          Recognition Engine Overview
        </h2>
        <span className="dashboard-card-badge">Engine Principles</span>
      </div>

      <div className="recognition-features-grid">
        {RECOGNITION_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <div key={item.title} className="feature-item">
              <div className="feature-icon-wrap" aria-hidden="true">
                <Icon size={18} />
              </div>
              <h3 className="feature-title">{item.title}</h3>
              <p className="feature-desc">{item.desc}</p>
            </div>
          );
        })}
      </div>
    </article>
  );
}
