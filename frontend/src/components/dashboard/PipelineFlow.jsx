import React from "react";
import {
  Camera,
  Scan,
  Sliders,
  Cpu,
  CheckCircle2,
  Volume2,
  ChevronRight,
  Info,
} from "lucide-react";

const PIPELINE_STEPS = [
  {
    step: 1,
    title: "Webcam / Video",
    desc: "Gesture input",
    icon: Camera,
  },
  {
    step: 2,
    title: "MediaPipe Landmarks",
    desc: "75 holistic keypoints",
    icon: Scan,
  },
  {
    step: 3,
    title: "Feature Engineering",
    desc: "Normalized 600 features",
    icon: Sliders,
  },
  {
    step: 4,
    title: "Bi-GRU Recognition",
    desc: "Temporal sequence (64)",
    icon: Cpu,
  },
  {
    step: 5,
    title: "Prediction",
    desc: "Top candidate + score",
    icon: CheckCircle2,
  },
  {
    step: 6,
    title: "Text & Speech",
    desc: "Audio synthesis",
    icon: Volume2,
  },
];

export default function PipelineFlow() {
  return (
    <article className="dashboard-card" aria-labelledby="how-it-works-title">
      <div className="dashboard-card-header">
        <h2 id="how-it-works-title" className="dashboard-card-title">
          How It Works
        </h2>
        <span className="dashboard-card-badge">Inference Pipeline</span>
      </div>

      <div className="pipeline-flow" role="list" aria-label="Inference pipeline steps">
        {PIPELINE_STEPS.map((step, index) => {
          const Icon = step.icon;
          return (
            <React.Fragment key={step.step}>
              <div className="pipeline-step" role="listitem">
                <div className="pipeline-step-icon" aria-hidden="true">
                  <Icon size={18} />
                </div>
                <div className="pipeline-step-title">{step.title}</div>
                <div className="pipeline-step-sub">{step.desc}</div>
              </div>

              {index < PIPELINE_STEPS.length - 1 && (
                <div className="pipeline-connector" aria-hidden="true">
                  <ChevronRight size={18} />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      <div className="pipeline-mvp-banner">
        <Info size={18} style={{ minWidth: "18px", marginTop: "2px" }} />
        <div>
          <strong>MVP Architecture Scope:</strong> The current system operates as an isolated sign recognizer using fixed 3-second capture windows with temporal alignment. Unrestricted continuous sentence translation is experimental and remains under development.
        </div>
      </div>
    </article>
  );
}
