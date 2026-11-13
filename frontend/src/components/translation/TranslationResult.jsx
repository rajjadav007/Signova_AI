import React from "react";
import SpeechSynthesizer from "../speech/SpeechSynthesizer";

/**
 * TranslationResult component.
 * Displays predicted text, model metadata, latency, decoding options, and TTS controls.
 */
export default function TranslationResult({
  result,
  loading = false,
  error = null,
  decoding = "beam",
  onDecodingChange,
  autoSpeak = true,
  onAutoSpeakChange,
}) {
  return (
    <div
      className="translation-card"
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
          Translation & Speech
        </h2>

        <div style={{ display: "flex", alignItems: "center", gap: "12px", fontSize: "0.85rem" }}>
          <label style={{ display: "flex", alignItems: "center", gap: "4px" }}>
            <span>Decoding:</span>
            <select
              value={decoding}
              onChange={(e) => onDecodingChange && onDecodingChange(e.target.value)}
              style={{ padding: "4px 8px", borderRadius: "4px", backgroundColor: "#333", color: "#fff", border: "1px solid #555" }}
            >
              <option value="beam">Beam Search (4 beams)</option>
              <option value="greedy">Greedy Search</option>
            </select>
          </label>

          <label style={{ display: "flex", alignItems: "center", gap: "4px", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={autoSpeak}
              onChange={(e) => onAutoSpeakChange && onAutoSpeakChange(e.target.checked)}
            />
            Auto-Speak
          </label>
        </div>
      </div>

      {/* Main Prediction Display */}
      <div
        style={{
          minHeight: "100px",
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
            <p style={{ margin: 0 }}>Extracting landmarks and running iSign T5 inference...</p>
          </div>
        ) : error ? (
          <div style={{ color: "#ef5350" }}>
            <strong>Error:</strong> {error}
          </div>
        ) : result && result.predicted_text ? (
          <div>
            <span style={{ fontSize: "0.8rem", color: "#888", textTransform: "uppercase" }}>
              Predicted English Text:
            </span>
            <p
              style={{
                fontSize: "1.6rem",
                fontWeight: "600",
                margin: "8px 0 0 0",
                color: "#e0e0e0",
                lineHeight: "1.3",
              }}
            >
              "{result.predicted_text}"
            </p>
          </div>
        ) : (
          <div style={{ textAlign: "center", color: "#777" }}>
            <p style={{ margin: 0 }}>Perform a sign gesture and click "Done Signing" to translate.</p>
          </div>
        )}
      </div>

      {/* Text-to-Speech Controls */}
      {result && result.predicted_text && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <SpeechSynthesizer text={result.predicted_text} autoSpeak={autoSpeak} />
          {result.inference_time_seconds !== undefined && (
            <div style={{ fontSize: "0.8rem", color: "#aaa" }}>
              Latency: {result.inference_time_seconds.toFixed(2)}s | Device: {result.device || "cpu"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
