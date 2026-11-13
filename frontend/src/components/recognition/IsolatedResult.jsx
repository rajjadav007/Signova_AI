import React from "react";
import SpeechSynthesizer from "../speech/SpeechSynthesizer";

/**
 * IsolatedResult component.
 * Displays recognized individual ISL sign, status pill, top-3 candidates, latency, and speech output.
 */
export default function IsolatedResult({
  result,
  loading = false,
  error = null,
  autoSpeak = true,
  onAutoSpeakChange,
}) {
  const isRecognized = result && result.status === "recognized";
  const isUncertain = result && result.status === "uncertain";

  return (
    <div
      className="isolated-result-card"
      style={{
        backgroundColor: "#222",
        color: "#f5f5f5",
        padding: "20px",
        borderRadius: "8px",
        boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
        display: "flex",
        flexDirection: "column",
        gap: "16px",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ margin: 0, fontSize: "1.25rem", color: "#64b5f6" }}>
          Word Recognition & Candidates
        </h2>

        <div style={{ display: "flex", alignItems: "center", gap: "12px", fontSize: "0.85rem" }}>
          <label style={{ display: "flex", alignItems: "center", gap: "6px", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={autoSpeak}
              onChange={(e) => onAutoSpeakChange && onAutoSpeakChange(e.target.checked)}
            />
            Auto-Speak Word
          </label>
        </div>
      </div>

      {/* Main Result Display Box */}
      <div
        style={{
          minHeight: "110px",
          backgroundColor: "#161616",
          borderRadius: "6px",
          padding: "16px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          border: "1px solid #333",
        }}
      >
        {loading ? (
          <div style={{ textAlign: "center", color: "#ffa726" }}>
            <p style={{ margin: 0, fontWeight: "500" }}>
              Extracting MediaPipe landmarks and classifying sign...
            </p>
          </div>
        ) : error ? (
          <div style={{ color: "#ef5350" }}>
            <strong>Error:</strong> {error}
          </div>
        ) : result && result.predicted_sign ? (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
              <span style={{ fontSize: "0.8rem", color: "#888", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                Recognized ISL Sign
              </span>
              <span
                style={{
                  fontSize: "0.75rem",
                  padding: "3px 8px",
                  borderRadius: "12px",
                  fontWeight: "600",
                  backgroundColor: isRecognized ? "#1b5e20" : "#e65100",
                  color: isRecognized ? "#a5d6a7" : "#ffe0b2",
                }}
              >
                {isRecognized ? "✓ RECOGNIZED" : "⚠ UNCERTAIN"}
              </span>
            </div>

            <p
              style={{
                fontSize: "2rem",
                fontWeight: "700",
                margin: "0 0 8px 0",
                color: isRecognized ? "#ffffff" : "#ffb74d",
                textTransform: "capitalize",
              }}
            >
              {result.display_name || result.predicted_sign}
            </p>

            {isUncertain && result.reason && (
              <p style={{ margin: "0 0 10px 0", fontSize: "0.82rem", color: "#ffb74d" }}>
                {result.reason}
              </p>
            )}

            {/* Top-3 Candidate Probabilities */}
            {result.top_predictions && result.top_predictions.length > 0 && (
              <div style={{ marginTop: "12px", borderTop: "1px solid #282828", paddingTop: "10px" }}>
                <span style={{ fontSize: "0.75rem", color: "#aaa", display: "block", marginBottom: "6px" }}>
                  Top Candidates:
                </span>
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  {result.top_predictions.map((cand, idx) => {
                    const pct = Math.round(cand.confidence * 100);
                    return (
                      <div key={idx} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.82rem" }}>
                        <span style={{ width: "90px", textTransform: "capitalize", color: idx === 0 ? "#fff" : "#888" }}>
                          {cand.sign}
                        </span>
                        <div style={{ flex: 1, backgroundColor: "#262626", borderRadius: "3px", height: "8px", overflow: "hidden" }}>
                          <div
                            style={{
                              width: `${pct}%`,
                              height: "100%",
                              backgroundColor: idx === 0 ? (isRecognized ? "#4caf50" : "#ffa726") : "#42a5f5",
                              transition: "width 0.3s ease",
                            }}
                          />
                        </div>
                        <span style={{ width: "38px", textAlign: "right", color: "#aaa" }}>
                          {pct}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div style={{ textAlign: "center", color: "#777" }}>
            <p style={{ margin: 0 }}>
              Show one of 10 supported signs (Hello, Thank you, Water, Food, Help, Please, Yes, No, Sorry, Stop) and click "Done Signing".
            </p>
          </div>
        )}
      </div>

      {/* TTS Speech Trigger and Latency Metadata */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.8rem", color: "#777" }}>
        <div>
          {result && result.predicted_sign && (
            <SpeechSynthesizer
              text={result.predicted_sign}
              autoSpeak={autoSpeak && isRecognized}
            />
          )}
        </div>

        {result && (
          <div style={{ textAlign: "right" }}>
            <span>Latency: {result.processing_time_ms ? `${result.processing_time_ms} ms` : "< 100 ms"}</span>
            {result.model_used && (
              <span style={{ marginLeft: "8px", color: "#555" }}>&bull; {result.model_used}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
