import React from "react";
import { BookOpen, Cpu, Target, Server, RefreshCw } from "lucide-react";
import { useBackendHealth } from "../../hooks/useBackendHealth";

export default function SystemOverviewCards() {
  const { status, data, isChecking, refetch } = useBackendHealth();

  return (
    <section aria-label="System Overview" className="overview-grid">
      {/* Card 1: Supported Signs */}
      <article className="overview-card" data-testid="card-supported-signs">
        <div className="overview-card-header">
          <span className="overview-card-label">Supported Signs</span>
          <div className="overview-card-icon" aria-hidden="true">
            <BookOpen size={18} />
          </div>
        </div>
        <div className="overview-card-value">10</div>
        <div className="overview-card-meta">ISL MVP Classes</div>
      </article>

      {/* Card 2: Recognition Model */}
      <article className="overview-card" data-testid="card-recognition-model">
        <div className="overview-card-header">
          <span className="overview-card-label">Recognition Model</span>
          <div className="overview-card-icon" aria-hidden="true">
            <Cpu size={18} />
          </div>
        </div>
        <div className="overview-card-value">Bi-GRU</div>
        <div className="overview-card-meta">2-Layer Bidirectional</div>
      </article>

      {/* Card 3: Recognition Type */}
      <article className="overview-card" data-testid="card-recognition-type">
        <div className="overview-card-header">
          <span className="overview-card-label">Recognition Type</span>
          <div className="overview-card-icon" aria-hidden="true">
            <Target size={18} />
          </div>
        </div>
        <div className="overview-card-value" style={{ fontSize: "1.25rem", marginTop: "4px" }}>
          Isolated Sign
        </div>
        <div className="overview-card-meta">Temporal Gesture Windows</div>
      </article>

      {/* Card 4: Backend Service */}
      <article className="overview-card" data-testid="card-backend-service">
        <div className="overview-card-header">
          <span className="overview-card-label">Backend Service</span>
          <div className="overview-card-icon" aria-hidden="true">
            <Server size={18} />
          </div>
        </div>
        <div className="overview-card-value">
          <div className={`backend-status-indicator ${status}`}>
            <span className="status-mini-dot" aria-hidden="true" />
            <span>
              {status === "connected"
                ? "Connected"
                : status === "disconnected"
                ? "Disconnected"
                : "Checking..."}
            </span>
            <button
              type="button"
              className="overview-refresh-btn"
              onClick={refetch}
              disabled={isChecking}
              title="Refresh backend status"
              aria-label="Refresh backend status"
            >
              <RefreshCw size={11} className={isChecking ? "rotating" : ""} />
              <span>Refresh</span>
            </button>
          </div>
        </div>
        <div className="overview-card-meta">
          {status === "connected" && data
            ? `${data.application || "Signova Backend"} (v${data.version || "0.1.0"})`
            : status === "disconnected"
            ? "API unreachable (port 8000)"
            : "Verifying endpoint health..."}
        </div>
      </article>
    </section>
  );
}
