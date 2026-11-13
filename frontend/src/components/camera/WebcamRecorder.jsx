import React, { useEffect, useRef, useState } from "react";

/**
 * WebcamRecorder component.
 * Allows capturing webcam video via MediaRecorder or selecting a pre-recorded sign video / .pose file.
 */
export default function WebcamRecorder({ onVideoReady, disabled = false }) {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);

  const [cameraActive, setCameraActive] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [cameraError, setCameraError] = useState(null);

  // Timer for recording duration
  useEffect(() => {
    let interval = null;
    if (recording) {
      interval = setInterval(() => {
        setRecordSeconds((s) => s + 1);
      }, 1000);
    } else {
      setRecordSeconds(0);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [recording]);

  // Clean up stream on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    setCameraError(null);
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("getUserMedia is not supported by your browser.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraActive(true);
    } catch (err) {
      setCameraError(err.message || "Failed to access webcam.");
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
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

  const startRecording = () => {
    if (!streamRef.current) return;
    chunksRef.current = [];
    setCameraError(null);

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

      recorder.onstop = () => {
        const recordedBlob = new Blob(chunksRef.current, { type: mimeType });
        const extension = mimeType.includes("mp4") ? "mp4" : "webm";
        if (onVideoReady) {
          onVideoReady(recordedBlob, `webcam_gesture.${extension}`);
        }
      };

      mediaRecorderRef.current = recorder;
      recorder.start(100); // 100ms slice
      setRecording(true);
    } catch (err) {
      setCameraError(`Failed to start recording: ${err.message}`);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files && e.target.files[0];
    if (file && onVideoReady) {
      onVideoReady(file, file.name);
    }
  };

  return (
    <div className="webcam-recorder" style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <div
        className="video-container"
        style={{
          width: "100%",
          maxWidth: "640px",
          height: "360px",
          backgroundColor: "#1a1a1a",
          borderRadius: "8px",
          overflow: "hidden",
          position: "relative",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          border: recording ? "2px solid #e53935" : "1px solid #444",
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
            display: cameraActive ? "block" : "none",
          }}
        />

        {!cameraActive && (
          <div style={{ color: "#888", textAlign: "center", padding: "16px" }}>
            <p>Camera is currently turned off.</p>
            <p style={{ fontSize: "0.85em" }}>Click "Start Camera" to enable webcam for sign recognition.</p>
          </div>
        )}

        {recording && (
          <div
            style={{
              position: "absolute",
              top: "12px",
              left: "12px",
              backgroundColor: "rgba(229, 57, 53, 0.85)",
              color: "white",
              padding: "4px 10px",
              borderRadius: "4px",
              fontSize: "0.85em",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <span
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                backgroundColor: "white",
              }}
            />
            Recording ({recordSeconds}s)
          </div>
        )}
      </div>

      {cameraError && (
        <div style={{ color: "#e53935", fontSize: "0.9em", padding: "6px 0" }}>
          ⚠️ {cameraError}
        </div>
      )}

      {/* Camera & Recording Controls */}
      <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "center" }}>
        {!cameraActive ? (
          <button
            type="button"
            onClick={startCamera}
            disabled={disabled}
            style={{
              padding: "8px 16px",
              backgroundColor: "#1976d2",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            Start Camera
          </button>
        ) : (
          <>
            <button
              type="button"
              onClick={stopCamera}
              disabled={recording || disabled}
              style={{
                padding: "8px 16px",
                backgroundColor: "#555",
                color: "white",
                border: "none",
                borderRadius: "4px",
                cursor: "pointer",
              }}
            >
              Stop Camera
            </button>

            {!recording ? (
              <button
                type="button"
                onClick={startRecording}
                disabled={disabled}
                style={{
                  padding: "8px 18px",
                  backgroundColor: "#e53935",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                  fontWeight: "bold",
                }}
              >
                Start Signing
              </button>
            ) : (
              <button
                type="button"
                onClick={stopRecording}
                disabled={disabled}
                style={{
                  padding: "8px 18px",
                  backgroundColor: "#2e7d32",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                  fontWeight: "bold",
                }}
              >
                Done Signing
              </button>
            )}
          </>
        )}

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "8px" }}>
          <label
            style={{
              fontSize: "0.9em",
              color: "#aaa",
              cursor: "pointer",
              textDecoration: "underline",
            }}
          >
            Or upload video/.pose file
            <input
              type="file"
              accept=".mp4,.mov,.avi,.webm,.mkv,.pose"
              onChange={handleFileUpload}
              disabled={disabled || recording}
              style={{ display: "none" }}
            />
          </label>
        </div>
      </div>
    </div>
  );
}
