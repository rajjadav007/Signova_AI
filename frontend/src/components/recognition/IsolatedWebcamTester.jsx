import React, { useEffect, useRef, useState } from "react";
import SpeechSynthesizer from "../speech/SpeechSynthesizer";
import { recognizeIsolatedVideo } from "../../services/api";

const ISOLATED_SIGNS = [
  "hello",
  "thank you",
  "sorry",
  "please",
  "yes",
  "no",
  "help",
  "stop",
  "water",
  "food",
];

const RECORDING_DURATION_MS = 3000; // 3 seconds optimal isolated gesture capture

export default function IsolatedWebcamTester() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const countdownIntervalRef = useRef(null);

  // Camera and recording states
  const [cameraActive, setCameraActive] = useState(false);
  const [recording, setRecording] = useState(false);
  const [countdown, setCountdown] = useState(3);
  const [cameraError, setCameraError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Prediction states
  const [result, setResult] = useState(null);
  const [expectedSign, setExpectedSign] = useState("");
  const [autoSpeak, setAutoSpeak] = useState(true);

  // Test history
  const [history, setHistory] = useState([]);

  // Cleanup stream on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
    };
  }, []);

  const startCamera = async () => {
    setCameraError(null);
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Camera API (getUserMedia) not supported by this browser.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
          frameRate: { ideal: 30 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraActive(true);
    } catch (err) {
      setCameraError(err.message || "Failed to access webcam. Please verify camera permissions.");
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
    if (mediaRecorderRef.current && recording) {
      try {
        mediaRecorderRef.current.stop();
      } catch (e) {
        // ignore
      }
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setRecording(false);
  };

  const captureAndRecognize = () => {
    if (!streamRef.current || recording || loading) return;

    setCameraError(null);
    chunksRef.current = [];
    setCountdown(3);

    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : MediaRecorder.isTypeSupported("video/webm")
      ? "video/webm"
      : "video/mp4";

    try {
      const recorder = new MediaRecorder(streamRef.current, { mimeType });
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        setRecording(false);
        if (chunksRef.current.length === 0) {
          setCameraError("Recording produced no video data. Please try again.");
          return;
        }

        const videoBlob = new Blob(chunksRef.current, { type: mimeType });
        const ext = mimeType.includes("mp4") ? "mp4" : "webm";
        await submitVideoForRecognition(videoBlob, `webcam_gesture.${ext}`);
      };

      mediaRecorderRef.current = recorder;
      recorder.start(100);
      setRecording(true);

      // Automated 3-second countdown
      let remaining = 3;
      countdownIntervalRef.current = setInterval(() => {
        remaining -= 1;
        setCountdown(remaining);
        if (remaining <= 0) {
          clearInterval(countdownIntervalRef.current);
          countdownIntervalRef.current = null;
          if (recorder.state === "recording") {
            recorder.stop();
          }
        }
      }, 1000);
    } catch (err) {
      setCameraError(`Recording failed: ${err.message}`);
      setRecording(false);
    }
  };

  const submitVideoForRecognition = async (fileBlob, fileName) => {
    setLoading(true);
    setCameraError(null);
    const testExpected = expectedSign; // snapshot for history
    try {
      const response = await recognizeIsolatedVideo(fileBlob, fileName, null, 3);
      setResult(response);

      // Record test history entry
      const isRecognized = response.status === "recognized";
      let matchResult = "EVALUATED";
      if (testExpected) {
        if (!isRecognized) {
          matchResult = "UNCERTAIN";
        } else if (response.predicted_sign.toLowerCase() === testExpected.toLowerCase()) {
          matchResult = "CORRECT";
        } else {
          matchResult = "INCORRECT";
        }
      } else {
        matchResult = isRecognized ? "RECOGNIZED" : "UNCERTAIN";
      }

      setHistory((prev) => [
        {
          id: prev.length + 1,
          time: new Date().toLocaleTimeString(),
          expected: testExpected || "None",
          predicted: response.predicted_sign,
          confidence: response.confidence,
          status: response.status,
          result: matchResult,
          latencyMs: response.processing_time_ms,
        },
        ...prev,
      ]);
    } catch (err) {
      setCameraError(err.message || "Recognition request failed. Please ensure the backend is running.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      submitVideoForRecognition(file, file.name);
    }
  };

  const resetHistory = () => {
    setHistory([]);
  };

  // History stats
  const totalTests = history.length;
  const correctTests = history.filter((h) => h.result === "CORRECT").length;
  const incorrectTests = history.filter((h) => h.result === "INCORRECT").length;
  const uncertainTests = history.filter((h) => h.status === "uncertain").length;

  const isRecognized = result && result.status === "recognized";
  const isUncertain = result && result.status === "uncertain";

  return (
    <div className="isolated-webcam-tester" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* Low-Data Warning Banner */}
      <div
        style={{
          backgroundColor: "#2a1c05",
          border: "1px solid #ff9800",
          borderRadius: "6px",
          padding: "10px 14px",
          fontSize: "0.85rem",
          color: "#ffe0b2",
          display: "flex",
          alignItems: "center",
          gap: "10px",
        }}
      >
        <span style={{ fontSize: "1.2rem" }}>⚠️</span>
        <div>
          <strong>Low-Volume Sign Notice:</strong> The sign <code>stop</code> (1 train sample) and <code>sorry</code> (3 train samples) have very limited training data and may be rejected as uncertain. The other 8 signs (<code>hello</code>, <code>thank you</code>, <code>please</code>, <code>help</code>, <code>yes</code>, <code>no</code>, <code>water</code>, <code>food</code>) are fully supported.
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
        {/* Left Column: Live Webcam Stream & Recording Controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <div
            style={{
              position: "relative",
              width: "100%",
              height: "360px",
              backgroundColor: "#111",
              borderRadius: "8px",
              overflow: "hidden",
              border: recording ? "3px solid #f44336" : "1px solid #444",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                transform: "scaleX(-1)", // Mirror preview for natural human signing feedback
                display: cameraActive ? "block" : "none",
              }}
            />

            {!cameraActive && (
              <div style={{ textAlign: "center", color: "#888", padding: "20px" }}>
                <p style={{ margin: "0 0 8px 0", fontSize: "1.1rem", fontWeight: "600" }}>Camera Inactive</p>
                <p style={{ margin: 0, fontSize: "0.85rem", color: "#666" }}>
                  Click "Start Camera" to enable live sign capture.
                </p>
              </div>
            )}

            {/* Signing Area Guide Box */}
            {cameraActive && !recording && (
              <div
                style={{
                  position: "absolute",
                  width: "70%",
                  height: "80%",
                  border: "2px dashed rgba(255,255,255,0.25)",
                  borderRadius: "12px",
                  pointerEvents: "none",
                  display: "flex",
                  alignItems: "flex-end",
                  justifyContent: "center",
                  paddingBottom: "10px",
                }}
              >
                <span style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.6)", backgroundColor: "rgba(0,0,0,0.5)", padding: "2px 8px", borderRadius: "4px" }}>
                  Keep hands & upper body in frame
                </span>
              </div>
            )}

            {/* Recording Active Overlay */}
            {recording && (
              <div
                style={{
                  position: "absolute",
                  top: "12px",
                  left: "12px",
                  right: "12px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div
                  style={{
                    backgroundColor: "rgba(244, 67, 54, 0.9)",
                    color: "white",
                    padding: "4px 12px",
                    borderRadius: "20px",
                    fontSize: "0.85rem",
                    fontWeight: "bold",
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                  }}
                >
                  <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: "white", animation: "pulse 1s infinite" }} />
                  RECORDING: {countdown}s remaining
                </div>
                <span style={{ fontSize: "0.8rem", color: "#fff", backgroundColor: "rgba(0,0,0,0.6)", padding: "3px 8px", borderRadius: "4px" }}>
                  Perform Sign Now
                </span>
              </div>
            )}

            {/* Processing Overlay */}
            {loading && (
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  backgroundColor: "rgba(0,0,0,0.75)",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#64b5f6",
                  gap: "10px",
                }}
              >
                <div style={{ fontSize: "1.1rem", fontWeight: "600" }}>Extracting Landmarks & Classifying...</div>
                <div style={{ fontSize: "0.85rem", color: "#aaa" }}>Bi-GRU Model Inference</div>
              </div>
            )}
          </div>

          {/* Camera Controls Bar */}
          <div style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}>
            {!cameraActive ? (
              <button
                type="button"
                onClick={startCamera}
                disabled={loading}
                style={{
                  padding: "10px 18px",
                  backgroundColor: "#2196f3",
                  color: "white",
                  border: "none",
                  borderRadius: "6px",
                  cursor: "pointer",
                  fontWeight: "600",
                }}
              >
                Start Camera
              </button>
            ) : (
              <>
                <button
                  type="button"
                  onClick={stopCamera}
                  disabled={recording || loading}
                  style={{
                    padding: "10px 16px",
                    backgroundColor: "#424242",
                    color: "#eee",
                    border: "none",
                    borderRadius: "6px",
                    cursor: "pointer",
                  }}
                >
                  Stop Camera
                </button>

                <button
                  type="button"
                  onClick={captureAndRecognize}
                  disabled={recording || loading}
                  style={{
                    padding: "10px 22px",
                    backgroundColor: recording ? "#f44336" : "#4caf50",
                    color: "white",
                    border: "none",
                    borderRadius: "6px",
                    cursor: "pointer",
                    fontWeight: "bold",
                    fontSize: "0.95rem",
                  }}
                >
                  {recording ? `Recording (${countdown}s)...` : "Capture & Recognize (3s)"}
                </button>
              </>
            )}

            <div style={{ marginLeft: "auto" }}>
              <label style={{ fontSize: "0.85rem", color: "#888", cursor: "pointer", textDecoration: "underline" }}>
                Upload test video (.mp4/.webm)
                <input
                  type="file"
                  accept=".mp4,.webm,.avi,.mov"
                  onChange={handleFileUpload}
                  disabled={loading || recording}
                  style={{ display: "none" }}
                />
              </label>
            </div>
          </div>

          {/* Expected Sign Selector (Test Mode) */}
          <div
            style={{
              backgroundColor: "#1a1a1a",
              border: "1px solid #333",
              borderRadius: "6px",
              padding: "10px 12px",
              display: "flex",
              alignItems: "center",
              gap: "12px",
            }}
          >
            <label style={{ fontSize: "0.85rem", color: "#aaa", whiteSpace: "nowrap" }}>
              Expected Sign (Optional Test Benchmark):
            </label>
            <select
              value={expectedSign}
              onChange={(e) => setExpectedSign(e.target.value)}
              disabled={loading}
              style={{
                flex: 1,
                padding: "6px 8px",
                backgroundColor: "#262626",
                color: "#fff",
                border: "1px solid #444",
                borderRadius: "4px",
                fontSize: "0.85rem",
              }}
            >
              <option value="">-- None / Unspecified (Blind Test) --</option>
              {ISOLATED_SIGNS.map((s) => (
                <option key={s} value={s}>
                  {s.toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {cameraError && (
            <div style={{ color: "#ef5350", fontSize: "0.85rem", backgroundColor: "rgba(239,83,80,0.1)", padding: "8px", borderRadius: "4px" }}>
              ⚠️ {cameraError}
            </div>
          )}
        </div>

        {/* Right Column: Prediction Results Area */}
        <div
          style={{
            backgroundColor: "#1e1e1e",
            borderRadius: "8px",
            border: "1px solid #333",
            padding: "20px",
            display: "flex",
            flexDirection: "column",
            gap: "16px",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3 style={{ margin: 0, fontSize: "1.1rem", color: "#90caf9" }}>Prediction Result</h3>
            <label style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.8rem", color: "#aaa", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={autoSpeak}
                onChange={(e) => setAutoSpeak(e.target.checked)}
              />
              Auto-Speak Result
            </label>
          </div>

          {/* Main Prediction Display Box */}
          <div
            style={{
              minHeight: "160px",
              backgroundColor: "#141414",
              borderRadius: "6px",
              border: "1px solid #292929",
              padding: "16px",
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
            }}
          >
            {loading ? (
              <div style={{ textAlign: "center", color: "#ffb74d" }}>
                <p style={{ margin: 0, fontWeight: "500" }}>Extracting 75 MediaPipe landmarks and classifying sequence...</p>
              </div>
            ) : result ? (
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                  <span style={{ fontSize: "0.75rem", color: "#888", letterSpacing: "1px" }}>
                    PREDICTED ISL SIGN
                  </span>
                  <span
                    style={{
                      fontSize: "0.75rem",
                      padding: "3px 10px",
                      borderRadius: "12px",
                      fontWeight: "bold",
                      backgroundColor: isRecognized ? "#1b5e20" : "#e65100",
                      color: isRecognized ? "#a5d6a7" : "#ffe0b2",
                    }}
                  >
                    {isRecognized ? "✓ RECOGNIZED" : "⚠ UNCERTAIN"}
                  </span>
                </div>

                <div style={{ display: "flex", alignItems: "baseline", gap: "14px" }}>
                  <span
                    style={{
                      fontSize: "2.4rem",
                      fontWeight: "800",
                      color: isRecognized ? "#fff" : "#ffb74d",
                      textTransform: "uppercase",
                    }}
                  >
                    {result.display_name || result.predicted_sign}
                  </span>
                  <span style={{ fontSize: "1.1rem", color: isRecognized ? "#81c784" : "#ffb74d", fontWeight: "600" }}>
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Uncertain Notice if Below Threshold */}
                {isUncertain && (
                  <div
                    style={{
                      marginTop: "10px",
                      padding: "8px 12px",
                      backgroundColor: "rgba(255, 152, 0, 0.15)",
                      borderLeft: "3px solid #ff9800",
                      borderRadius: "4px",
                      fontSize: "0.85rem",
                      color: "#ffe0b2",
                    }}
                  >
                    <strong>Sign not recognized confidently. Please try again.</strong>
                    {result.reason && <div style={{ fontSize: "0.78rem", color: "#ffb74d", marginTop: "3px" }}>{result.reason}</div>}
                  </div>
                )}

                {/* Top-3 Candidates Breakdown */}
                {result.top_predictions && result.top_predictions.length > 0 && (
                  <div style={{ marginTop: "14px", borderTop: "1px solid #222", paddingTop: "10px" }}>
                    <span style={{ fontSize: "0.75rem", color: "#777", display: "block", marginBottom: "6px" }}>
                      Candidate Confidence Scores:
                    </span>
                    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                      {result.top_predictions.map((p, idx) => {
                        const pct = Math.round(p.confidence * 100);
                        const signName = p.label || p.sign || p.display_name;
                        return (
                          <div key={idx} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.8rem" }}>
                            <span style={{ width: "90px", textTransform: "capitalize", color: idx === 0 ? "#fff" : "#888" }}>
                              {signName}
                            </span>
                            <div style={{ flex: 1, height: "6px", backgroundColor: "#222", borderRadius: "3px", overflow: "hidden" }}>
                              <div
                                style={{
                                  width: `${pct}%`,
                                  height: "100%",
                                  backgroundColor: idx === 0 ? (isRecognized ? "#4caf50" : "#ff9800") : "#2196f3",
                                }}
                              />
                            </div>
                            <span style={{ width: "36px", textAlign: "right", color: "#888" }}>{pct}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ textAlign: "center", color: "#666", padding: "16px" }}>
                <p style={{ margin: "0 0 6px 0" }}>No sign captured yet.</p>
                <p style={{ margin: 0, fontSize: "0.82rem" }}>
                  Start Camera, prepare your sign, and click "Capture & Recognize".
                </p>
              </div>
            )}
          </div>

          {/* Model & Latency Footer */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.78rem", color: "#666" }}>
            <span>
              Model: {result?.model_used || "Bi-GRU (Bilateral Improved)"} &bull; Latency: {result ? `${result.processing_time_ms} ms` : "--"}
            </span>
            {result?.predicted_sign && isRecognized && (
              <SpeechSynthesizer text={result.predicted_sign} autoSpeak={autoSpeak} />
            )}
          </div>
        </div>
      </div>

      {/* Test History Section */}
      <div
        style={{
          backgroundColor: "#1a1a1a",
          borderRadius: "8px",
          border: "1px solid #333",
          padding: "16px 20px",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
          <div>
            <h4 style={{ margin: 0, fontSize: "1rem", color: "#eee" }}>Test History & Metrics</h4>
            <span style={{ fontSize: "0.8rem", color: "#888" }}>
              Total: {totalTests} | Correct: {correctTests} ({totalTests ? Math.round((correctTests / totalTests) * 100) : 0}%) | Incorrect: {incorrectTests} | Uncertain: {uncertainTests}
            </span>
          </div>
          <button
            type="button"
            onClick={resetHistory}
            disabled={history.length === 0}
            style={{
              padding: "6px 12px",
              backgroundColor: "#333",
              color: "#ccc",
              border: "1px solid #444",
              borderRadius: "4px",
              fontSize: "0.8rem",
              cursor: history.length === 0 ? "not-allowed" : "pointer",
            }}
          >
            Reset History
          </button>
        </div>

        {history.length > 0 ? (
          <div style={{ maxHeight: "220px", overflowY: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem", textAlign: "left" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #333", color: "#888" }}>
                  <th style={{ padding: "6px" }}>#</th>
                  <th style={{ padding: "6px" }}>Time</th>
                  <th style={{ padding: "6px" }}>Expected</th>
                  <th style={{ padding: "6px" }}>Predicted</th>
                  <th style={{ padding: "6px" }}>Confidence</th>
                  <th style={{ padding: "6px" }}>Status</th>
                  <th style={{ padding: "6px" }}>Result</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h) => {
                  let badgeColor = "#555";
                  if (h.result === "CORRECT") badgeColor = "#2e7d32";
                  if (h.result === "INCORRECT") badgeColor = "#c62828";
                  if (h.result === "UNCERTAIN") badgeColor = "#e65100";

                  return (
                    <tr key={h.id} style={{ borderBottom: "1px solid #222" }}>
                      <td style={{ padding: "6px", color: "#666" }}>{h.id}</td>
                      <td style={{ padding: "6px", color: "#888" }}>{h.time}</td>
                      <td style={{ padding: "6px", textTransform: "capitalize" }}>{h.expected}</td>
                      <td style={{ padding: "6px", fontWeight: "600", textTransform: "capitalize" }}>{h.predicted}</td>
                      <td style={{ padding: "6px" }}>{(h.confidence * 100).toFixed(1)}%</td>
                      <td style={{ padding: "6px" }}>{h.status}</td>
                      <td style={{ padding: "6px" }}>
                        <span
                          style={{
                            padding: "2px 8px",
                            borderRadius: "10px",
                            fontSize: "0.72rem",
                            fontWeight: "bold",
                            backgroundColor: badgeColor,
                            color: "white",
                          }}
                        >
                          {h.result}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ textAlign: "center", color: "#555", fontSize: "0.85rem", padding: "12px 0" }}>
            No tests recorded yet in this session.
          </div>
        )}
      </div>
    </div>
  );
}
