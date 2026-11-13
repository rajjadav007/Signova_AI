import React, { useState, useEffect, useRef, useMemo } from "react";
import PageContainer from "../components/common/PageContainer";
import {
  fetchDemoSentenceClasses,
  recognizeDemoSentenceVideo,
  fetchFiveClassClasses,
  recognizeFiveClassSentenceVideo,
} from "../services/api";
import {
  Upload,
  Video,
  CheckCircle,
  AlertTriangle,
  Play,
  Square,
  RotateCcw,
  Sparkles,
  BarChart3,
  Clock,
  Check,
  X,
  FileVideo,
  Layers,
  HelpCircle,
  Camera,
  Activity,
  Sliders,
} from "lucide-react";
import "../styles/sentenceTesting.css";

const FALLBACK_DEMO_CLASSES = [
  { demo_class_id: 0, original_sentence_id: 5, sentence: "comb your hair", isl_gloss: "COMB YOU HAIR" },
  { demo_class_id: 1, original_sentence_id: 23, sentence: "help me", isl_gloss: "HELP ME" },
  { demo_class_id: 2, original_sentence_id: 35, sentence: "i am fine. thank you sir", isl_gloss: "I FINE. THANK YOU SIR" },
  { demo_class_id: 3, original_sentence_id: 53, sentence: "i need water", isl_gloss: "I NEED WATER" },
  { demo_class_id: 4, original_sentence_id: 91, sentence: "who are you", isl_gloss: "WHO YOU" },
];

const ALLOWED_EXTENSIONS = [".mp4", ".webm", ".avi", ".mov", ".mkv"];
const MAX_FILE_SIZE_MB = 50;

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || isNaN(seconds)) return "--:--";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
}

export default function SentenceTestingPage() {
  const [demoClasses, setDemoClasses] = useState(FALLBACK_DEMO_CLASSES);

  // Model & Input Mode selection
  const [selectedModel, setSelectedModel] = useState("bigru"); // "bigru" | "dtw"
  const [inputMode, setInputMode] = useState("upload"); // "upload" | "webcam"

  // File upload state
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [videoDuration, setVideoDuration] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // Live Webcam state
  const [webcamActive, setWebcamActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [cameraError, setCameraError] = useState(null);

  // Inference state
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);

  // Manual verification state
  const [actualSentence, setActualSentence] = useState("");
  const [verifiedItem, setVerifiedItem] = useState(null);

  // Session history state
  const [sessionHistory, setSessionHistory] = useState([]);

  const fileInputRef = useRef(null);
  const webcamVideoRef = useRef(null);
  const webcamStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);

  // Load supported classes on mount from authoritative backend
  useEffect(() => {
    let isMounted = true;
    fetchFiveClassClasses()
      .then((data) => {
        if (isMounted && data && Array.isArray(data.classes) && data.classes.length > 0) {
          setDemoClasses(data.classes);
        }
      })
      .catch(() => {
        return fetchDemoSentenceClasses().then((data) => {
          if (isMounted && data && Array.isArray(data.classes) && data.classes.length > 0) {
            setDemoClasses(data.classes);
          }
        });
      })
      .catch((err) => {
        console.warn("Could not fetch remote demo classes, using fallback:", err);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  // Timer for live webcam recording duration
  useEffect(() => {
    let interval = null;
    if (isRecording) {
      interval = setInterval(() => {
        setRecordingSeconds((s) => s + 1);
      }, 1000);
    } else {
      setRecordingSeconds(0);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRecording]);

  // Clean up webcam stream on unmount
  useEffect(() => {
    return () => {
      if (webcamStreamRef.current) {
        webcamStreamRef.current.getTracks().forEach((track) => track.stop());
        webcamStreamRef.current = null;
      }
    };
  }, []);

  // Handle file selection
  const handleFileChange = (file) => {
    if (!file) return;

    setErrorMessage(null);
    setPredictionResult(null);
    setVerifiedItem(null);
    setActualSentence("");
    setVideoDuration(null);

    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setErrorMessage(
        `Unsupported video format "${ext}". Supported formats: ${ALLOWED_EXTENSIONS.join(", ")}`
      );
      return;
    }

    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setErrorMessage(
        `File size (${(file.size / (1024 * 1024)).toFixed(1)}MB) exceeds maximum limit of ${MAX_FILE_SIZE_MB}MB.`
      );
      return;
    }

    setSelectedFile(file);

    // Clean up previous blob URL
    if (videoPreviewUrl) {
      try {
        if (typeof window !== "undefined" && window.URL && window.URL.revokeObjectURL) {
          window.URL.revokeObjectURL(videoPreviewUrl);
        } else if (typeof URL !== "undefined" && URL.revokeObjectURL) {
          URL.revokeObjectURL(videoPreviewUrl);
        }
      } catch (_) {}
    }

    try {
      let previewUrl = "";
      if (typeof window !== "undefined" && window.URL && typeof window.URL.createObjectURL === "function") {
        previewUrl = window.URL.createObjectURL(file);
      } else if (typeof URL !== "undefined" && typeof URL.createObjectURL === "function") {
        previewUrl = URL.createObjectURL(file);
      }
      if (!previewUrl) {
        previewUrl = "blob:video-preview-mock";
      }
      setVideoPreviewUrl(previewUrl);

      // Probe duration if in browser environment
      if (typeof document !== "undefined" && previewUrl.startsWith("blob:")) {
        const tempVideo = document.createElement("video");
        tempVideo.preload = "metadata";
        tempVideo.onloadedmetadata = () => {
          if (tempVideo.duration && !isNaN(tempVideo.duration)) {
            setVideoDuration(tempVideo.duration);
          }
        };
        tempVideo.src = previewUrl;
      }
    } catch (_) {
      setVideoPreviewUrl("blob:video-preview-mock");
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setVideoDuration(null);
    if (videoPreviewUrl) {
      try {
        if (typeof window !== "undefined" && window.URL && window.URL.revokeObjectURL) {
          window.URL.revokeObjectURL(videoPreviewUrl);
        } else if (typeof URL !== "undefined" && URL.revokeObjectURL) {
          URL.revokeObjectURL(videoPreviewUrl);
        }
      } catch (_) {}
      setVideoPreviewUrl(null);
    }
    setPredictionResult(null);
    setVerifiedItem(null);
    setActualSentence("");
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Run recognition for uploaded video file
  const handleRecognize = async () => {
    if (!selectedFile || isLoading) return;

    setIsLoading(true);
    setErrorMessage(null);
    setPredictionResult(null);
    setVerifiedItem(null);

    try {
      const result = await recognizeFiveClassSentenceVideo(selectedFile, selectedModel);
      setPredictionResult(result);
    } catch (err) {
      console.error("Recognition error:", err);
      setErrorMessage(err.message || "Failed to process video with backend.");
    } finally {
      setIsLoading(false);
    }
  };

  // Live Webcam controls
  const startCamera = async () => {
    setCameraError(null);
    try {
      if (typeof navigator === "undefined" || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("getUserMedia is not supported by your browser or environment.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      webcamStreamRef.current = stream;
      if (webcamVideoRef.current) {
        webcamVideoRef.current.srcObject = stream;
      }
      setWebcamActive(true);
    } catch (err) {
      console.error("Camera access error:", err);
      setCameraError(err.message || "Failed to access webcam. Please check camera permissions.");
      setWebcamActive(false);
    }
  };

  const stopCamera = () => {
    if (mediaRecorderRef.current && isRecording) {
      try {
        mediaRecorderRef.current.stop();
      } catch (_) {}
      setIsRecording(false);
    }
    if (webcamStreamRef.current) {
      webcamStreamRef.current.getTracks().forEach((track) => track.stop());
      webcamStreamRef.current = null;
    }
    if (webcamVideoRef.current) {
      webcamVideoRef.current.srcObject = null;
    }
    setWebcamActive(false);
    setIsRecording(false);
  };

  const startSigning = () => {
    if (!webcamStreamRef.current) return;
    recordedChunksRef.current = [];
    setCameraError(null);
    setErrorMessage(null);
    setPredictionResult(null);
    setVerifiedItem(null);

    try {
      let mimeType = "video/webm";
      if (typeof MediaRecorder !== "undefined" && typeof MediaRecorder.isTypeSupported === "function") {
        if (MediaRecorder.isTypeSupported("video/webm;codecs=vp8")) {
          mimeType = "video/webm;codecs=vp8";
        } else if (MediaRecorder.isTypeSupported("video/webm")) {
          mimeType = "video/webm";
        } else if (MediaRecorder.isTypeSupported("video/mp4")) {
          mimeType = "video/mp4";
        }
      }

      const recorder = new MediaRecorder(webcamStreamRef.current, { mimeType });
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const recordedBlob = new Blob(recordedChunksRef.current, { type: mimeType });
        await processWebcamRecording(recordedBlob, mimeType);
      };

      recorder.start(100);
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch (err) {
      console.error("Recording error:", err);
      setCameraError(err.message || "Failed to start MediaRecorder on webcam stream.");
      setIsRecording(false);
    }
  };

  const stopSigning = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processWebcamRecording = async (blob, mimeType) => {
    if (!blob || blob.size === 0) {
      setErrorMessage("No video frames were captured during recording.");
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    setPredictionResult(null);
    setVerifiedItem(null);

    const ext = mimeType && mimeType.includes("mp4") ? "mp4" : "webm";
    const file = new File([blob], `webcam_signing.${ext}`, { type: mimeType || "video/webm" });

    try {
      const result = await recognizeFiveClassSentenceVideo(file, selectedModel);
      setPredictionResult(result);
    } catch (err) {
      console.error("Webcam recognition inference error:", err);
      setErrorMessage(err.message || "Failed to process webcam recording with backend.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetryRecording = () => {
    setPredictionResult(null);
    setErrorMessage(null);
    setVerifiedItem(null);
    setActualSentence("");
  };

  // Record manual verification and save to history (no duplicates)
  const handleVerify = () => {
    if (!predictionResult || !actualSentence) return;

    const isRecognized = predictionResult.is_recognized;
    const predicted = predictionResult.predicted_sentence;

    let correctness = "incorrect";
    if (actualSentence === "other_unknown") {
      correctness = !isRecognized ? "correct" : "incorrect";
    } else {
      correctness = (isRecognized && predicted === actualSentence) ? "correct" : "incorrect";
    }

    const uniqueKey = `${selectedFile ? selectedFile.name : `webcam_${selectedModel}`}_${predictionResult.processing_time_ms}`;
    const now = new Date();

    const verificationRecord = {
      id: Date.now(),
      key: uniqueKey,
      filename: selectedFile ? selectedFile.name : `webcam_${selectedModel}_signing.webm`,
      actual: actualSentence === "other_unknown" ? "Out of Vocabulary (Unknown)" : actualSentence,
      predicted: predicted ? (isRecognized ? predicted : `${predicted} (Low Conf)`) : "Sentence not recognized.",
      isRecognized,
      correctness,
      dtwDistance: predictionResult.top_distance ?? 0.0,
      processingTimeMs: predictionResult.processing_time_ms,
      timestamp: now.toLocaleTimeString(),
      dateTime: now.toLocaleString(),
    };

    setVerifiedItem(verificationRecord);

    setSessionHistory((prev) => {
      if (prev.some((item) => item.key === uniqueKey)) {
        return prev;
      }
      return [verificationRecord, ...prev];
    });
  };

  // Compute live session statistics
  const sessionStats = useMemo(() => {
    const total = sessionHistory.length;
    const correct = sessionHistory.filter((h) => h.correctness === "correct").length;
    const incorrect = sessionHistory.filter((h) => h.correctness === "incorrect").length;
    const rejected = sessionHistory.filter((h) => !h.isRecognized).length;
    const accuracy = total > 0 ? ((correct / total) * 100).toFixed(1) : "0.0";

    return { total, correct, incorrect, rejected, accuracy };
  }, [sessionHistory]);

  const clearHistory = () => {
    setSessionHistory([]);
    setVerifiedItem(null);
  };

  return (
    <PageContainer>
      <div className="sentence-testing-container" data-testid="sentence-testing-page">
        {/* Page Header */}
        <header className="testing-header">
          <div className="testing-title-group">
            <h1 data-testid="sentence-page-title">
              <Sparkles size={24} className="brand-accent" />
              Signova AI &bull; ISL Sentence Recognition
              <span className="project-tag">Hackathon Demo</span>
            </h1>
            <h2
              style={{
                position: "absolute",
                width: "1px",
                height: "1px",
                padding: 0,
                margin: "-1px",
                overflow: "hidden",
                clip: "rect(0, 0, 0, 0)",
                whiteSpace: "nowrap",
                border: 0,
              }}
            >
              Testing &amp; History
            </h2>
            <p className="testing-subtitle">
              Recognize continuous Indian Sign Language sentences via Live Webcam or Video Upload.
            </p>
          </div>
          <div className="header-badges">
            <span className="badge-tech">
              <CheckCircle size={14} color="var(--color-status-green)" />
              Model: Five-Class DTW-KNN &amp; BiGRU Attention
            </span>
            <span className="badge-tech" data-testid="active-model-badge">
              <Sparkles size={14} color="var(--color-primary)" />
              Active Engine: {selectedModel === "bigru" ? "BiGRU Attention" : "DTW-KNN"}
            </span>
            <span className="badge-tech">
              <Layers size={14} />
              Supported sentences: 5
            </span>
            <span className="badge-tech">
              <Clock size={14} />
              Recognition status: {isLoading ? "Processing..." : isRecording ? "Recording..." : "Ready"}
            </span>
          </div>
        </header>

        {/* Model Architecture Selector Toolbar */}
        <section className="model-selector-container" aria-label="Model Architecture Selection">
          <div className="model-selector-header">
            <span className="model-selector-title">
              <Sliders size={16} className="brand-accent" />
              Select Recognition Architecture
            </span>
            <span className="model-selector-hint">Choose between neural sequence modeling and temporal alignment</span>
          </div>
          <div className="model-toggle-group" data-testid="model-selector-group">
            <button
              type="button"
              className={`model-toggle-btn ${selectedModel === "bigru" ? "active" : ""}`}
              onClick={() => {
                setSelectedModel("bigru");
                setPredictionResult(null);
                setErrorMessage(null);
              }}
              data-testid="model-btn-bigru"
            >
              <Sparkles size={16} />
              <div className="model-btn-details">
                <span className="model-name">BiGRU Attention (Neural)</span>
                <span className="model-desc">Softmax probabilities &bull; 130K params &bull; 0.95ms inference</span>
              </div>
            </button>
            <button
              type="button"
              className={`model-toggle-btn ${selectedModel === "dtw" ? "active" : ""}`}
              onClick={() => {
                setSelectedModel("dtw");
                setPredictionResult(null);
                setErrorMessage(null);
              }}
              data-testid="model-btn-dtw"
            >
              <Layers size={16} />
              <div className="model-btn-details">
                <span className="model-name">DTW-KNN (Temporal Alignment)</span>
                <span className="model-desc">Cosine distance &bull; Template matching &bull; 5.4ms latency</span>
              </div>
            </button>
          </div>
        </section>

        {/* Supported Sentence Examples */}
        <section className="supported-sentences-section" aria-label="Supported Sentence Classes">
          <div className="supported-sentences-header">
            <h2>
              <CheckCircle size={18} color="var(--color-primary)" />
              Supported Sentences (5 Classes)
            </h2>
            <span className="sentence-pattern-pill">Full Sentence Patterns (Not Isolated Signs)</span>
          </div>
          <div className="supported-sentences-grid">
            {demoClasses.map((item, idx) => (
              <div key={item.demo_class_id ?? idx} className="supported-sentence-card">
                <span className="sentence-card-number">Sentence {idx + 1}</span>
                <span className="sentence-card-title">"{item.sentence}"</span>
                <span className="sentence-card-gloss">Gloss: {item.isl_gloss}</span>
              </div>
            ))}
          </div>
          <p className="sentence-explanatory-banner">
            <HelpCircle size={14} style={{ display: "inline", marginRight: "6px", verticalAlign: "middle" }} />
            These are complete continuous Indian Sign Language sentence patterns with MediaPipe landmark trajectories, not individual isolated signs.
          </p>
        </section>

        {/* Error Alert Box */}
        {(errorMessage || cameraError) && (
          <div className="error-alert-box" role="alert" data-testid="error-alert">
            <AlertTriangle size={18} />
            <span>{errorMessage || cameraError}</span>
          </div>
        )}

        {/* Main 2-Column Grid: Input Section & Prediction Result */}
        <div className="testing-main-grid">
          {/* Column A: Input Section (File Upload OR Live Webcam) */}
          <section className="testing-card" aria-label="Capture or Upload ISL Video">
            <div className="input-mode-tabs" data-testid="input-mode-tabs">
              <button
                type="button"
                className={`mode-tab-btn ${inputMode === "upload" ? "active" : ""}`}
                onClick={() => {
                  setInputMode("upload");
                  if (webcamActive) stopCamera();
                }}
                data-testid="tab-upload"
              >
                <Upload size={15} />
                Video File Upload
              </button>
              <button
                type="button"
                className={`mode-tab-btn ${inputMode === "webcam" ? "active" : ""}`}
                onClick={() => setInputMode("webcam")}
                data-testid="tab-webcam"
              >
                <Camera size={15} />
                Live Webcam Capture
              </button>
            </div>

            {inputMode === "upload" ? (
              <>
                <div className="card-header-clean">
                  <h2 className="card-title-clean">
                    <Video size={18} />
                    Video Upload &amp; Preview
                  </h2>
                  {selectedFile && (
                    <div className="preview-control-buttons">
                      <button
                        type="button"
                        className="btn-secondary-action"
                        onClick={() => fileInputRef.current && fileInputRef.current.click()}
                        title="Replace video"
                        style={{ padding: "4px 8px", fontSize: "0.8rem" }}
                      >
                        <RotateCcw size={13} style={{ marginRight: "4px" }} />
                        Replace
                      </button>
                      <button
                        type="button"
                        className="btn-secondary-action btn-danger-text"
                        onClick={handleClear}
                        title="Remove video"
                        style={{ padding: "4px 8px", fontSize: "0.8rem" }}
                      >
                        <X size={13} style={{ marginRight: "4px" }} />
                        Remove
                      </button>
                    </div>
                  )}
                </div>

                {/* Hidden File Input */}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".mp4,.webm,.avi,.mov,.mkv"
                  style={{ display: "none" }}
                  data-testid="sentence-file-input"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      handleFileChange(file);
                    }
                  }}
                />

                {/* Drag and Drop Upload Area */}
                {!videoPreviewUrl ? (
                  <div
                    className={`upload-dropzone ${isDragging ? "drag-active" : ""}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current && fileInputRef.current.click()}
                    data-testid="video-dropzone"
                  >
                    <div className="upload-dropzone-inner">
                      <div className="upload-icon-circle">
                        <Upload size={22} />
                      </div>
                      <span className="upload-text-main">Click to browse or drop ISL video here</span>
                      <span className="upload-text-sub">
                        Supported formats: MP4, WebM, AVI, MOV, MKV (Max 50MB)
                      </span>
                      <button
                        type="button"
                        className="btn-secondary-action"
                        style={{ marginTop: "8px", pointerEvents: "none" }}
                      >
                        Browse Files
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="video-preview-wrapper" data-testid="video-preview-wrapper">
                    <video
                      src={videoPreviewUrl}
                      controls
                      className="video-preview-player"
                      data-testid="video-preview-player"
                      onLoadedMetadata={(e) => {
                        if (e.target.duration && !isNaN(e.target.duration)) {
                          setVideoDuration(e.target.duration);
                        }
                      }}
                    />
                  </div>
                )}

                {/* File Info Bar */}
                {selectedFile && (
                  <div className="file-info-bar">
                    <span className="file-name-tag">
                      <FileVideo size={14} style={{ display: "inline", marginRight: "6px" }} />
                      {selectedFile.name}
                    </span>
                    <div className="file-meta-tags">
                      <span>Duration: {formatDuration(videoDuration)}</span>
                      <span>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                    </div>
                  </div>
                )}

                {/* Recognition Button */}
                <div className="action-buttons-group">
                  <button
                    type="button"
                    className="btn-primary-action"
                    onClick={handleRecognize}
                    disabled={!selectedFile || isLoading}
                    data-testid="btn-recognize-sentence"
                  >
                    {isLoading ? (
                      <>
                        <RotateCcw size={16} className="spin-animation" />
                        Extracting Landmarks &amp; Recognizing ({selectedModel === "bigru" ? "BiGRU" : "DTW"})...
                      </>
                    ) : (
                      <>
                        <Play size={16} />
                        Recognize Sentence ({selectedModel === "bigru" ? "BiGRU" : "DTW"})
                      </>
                    )}
                  </button>
                </div>
              </>
            ) : (
              /* Live Webcam Mode */
              <div className="webcam-mode-container" data-testid="webcam-section">
                {!webcamActive ? (
                  <div className="webcam-inactive-box" data-testid="webcam-inactive-box">
                    <Camera size={44} strokeWidth={1.5} className="webcam-icon-muted" />
                    <h3>Live Webcam Recognition</h3>
                    <p>
                      Sign an Indian Sign Language sentence in front of your camera. Chronological RGB frames will be processed using MediaPipe and evaluated with the <strong>{selectedModel === "bigru" ? "BiGRU Attention" : "DTW-KNN"}</strong> model.
                    </p>
                    <button
                      type="button"
                      className="btn-primary-action"
                      onClick={startCamera}
                      data-testid="btn-start-camera"
                    >
                      <Camera size={16} />
                      Start Camera Feed
                    </button>
                  </div>
                ) : (
                  <div className="webcam-active-view">
                    <div className="webcam-video-container">
                      <video
                        ref={webcamVideoRef}
                        autoPlay
                        playsInline
                        muted
                        className="webcam-video-player"
                        data-testid="webcam-video-feed"
                      />

                      {/* Live Tracking & Recording Overlays */}
                      <div className="webcam-overlay-bar">
                        <span className="tracking-status-badge">
                          <Activity size={12} color="var(--color-status-green)" />
                          75 Landmarks (Pose + Hands) Ready
                        </span>
                        {isRecording && (
                          <div className="recording-status-badge" data-testid="recording-indicator">
                            <span className="pulse-dot" />
                            Recording: {formatDuration(recordingSeconds)}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Camera Control Actions */}
                    <div className="webcam-controls-row">
                      {!isRecording ? (
                        <button
                          type="button"
                          className="btn-primary-action btn-record-start"
                          onClick={startSigning}
                          disabled={isLoading}
                          data-testid="btn-start-signing"
                        >
                          <Play size={16} />
                          Start Signing
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="btn-danger-action btn-record-stop"
                          onClick={stopSigning}
                          data-testid="btn-stop-signing"
                        >
                          <Square size={16} />
                          Stop Signing &amp; Recognize
                        </button>
                      )}

                      <button
                        type="button"
                        className="btn-secondary-action"
                        onClick={stopCamera}
                        disabled={isRecording || isLoading}
                        data-testid="btn-stop-camera"
                      >
                        Turn Off Camera
                      </button>
                    </div>

                    <p className="webcam-instruction-note">
                      {isRecording
                        ? "Perform the full ISL sentence naturally. Click 'Stop Signing & Recognize' when finished."
                        : "Click 'Start Signing', perform one of the 5 supported sentences, then click Stop."}
                    </p>
                  </div>
                )}
              </div>
            )}
          </section>

          {/* Column B: Recognition Results */}
          <section className="testing-card" aria-label="Recognition Results">
            <div className="card-header-clean">
              <h2 className="card-title-clean">
                <BarChart3 size={18} />
                Prediction Result
              </h2>
              {predictionResult && (
                <span
                  className={`status-tag ${predictionResult.is_recognized ? "success" : "rejected"}`}
                  data-testid="recognition-status-badge"
                >
                  {predictionResult.status}
                </span>
              )}
            </div>

            {!predictionResult ? (
              <div className="empty-placeholder-box">
                <Video size={36} strokeWidth={1.5} />
                <p>
                  {inputMode === "upload"
                    ? 'Upload an ISL sentence video and click "Recognize Sentence" to view prediction.'
                    : 'Start camera, click "Start Signing", perform an ISL sentence, and stop to view prediction.'}
                </p>
              </div>
            ) : (
              <>
                {/* Hero Prediction Card */}
                <div
                  className={`prediction-hero-card ${
                    predictionResult.is_recognized ? "recognized" : "rejected"
                  }`}
                  data-testid="prediction-hero"
                >
                  <div className="prediction-status-row">
                    <span className="hero-gloss">
                      {predictionResult.is_recognized
                        ? "Prediction"
                        : "Low confidence"}
                      {predictionResult.predicted_class_id !== null && predictionResult.predicted_class_id !== undefined
                        ? ` • Class ${predictionResult.predicted_class_id}${predictionResult.original_sentence_id ? ` (SID ${predictionResult.original_sentence_id})` : ""}`
                        : ""}
                    </span>
                    <span
                      style={{
                        fontSize: "0.8rem",
                        fontWeight: 600,
                        color: predictionResult.is_recognized
                          ? "var(--color-status-green)"
                          : "var(--color-status-yellow, #D97706)",
                      }}
                    >
                      {predictionResult.confidence !== undefined
                        ? `Prob: ${(predictionResult.confidence * 100).toFixed(1)}%`
                        : predictionResult.top_similarity !== undefined
                        ? `Sim: ${(predictionResult.top_similarity * 100).toFixed(1)}%`
                        : predictionResult.is_recognized
                        ? "Recognized"
                        : "Tentative"}
                    </span>
                  </div>

                  <h3 className="hero-sentence" data-testid="predicted-sentence-text">
                    {predictionResult.predicted_sentence
                      ? `"${predictionResult.predicted_sentence}"`
                      : "Sentence not recognized."}
                  </h3>
                </div>

                {/* Explanation if Rejected */}
                {!predictionResult.is_recognized && (
                  <div className="rejection-explanation-box">
                    <strong>Notice (Low Confidence):</strong>
                    <span>
                      {predictionResult.reason ||
                        "The model confidence is below the acceptance threshold. The tentative top candidate is displayed above along with all five class probabilities."}
                    </span>
                  </div>
                )}

                {/* Metrics Chips Grid */}
                <div className="metrics-chips-grid">
                  <div className="metric-chip">
                    <span className="metric-chip-label">Recognition Status</span>
                    <span
                      className="metric-chip-value"
                      style={{
                        fontSize: "0.95rem",
                        color: predictionResult.is_recognized
                          ? "var(--color-status-green)"
                          : "var(--color-status-yellow)",
                      }}
                    >
                      {predictionResult.is_recognized ? "Recognized" : "Unknown"}
                    </span>
                  </div>

                  <div className="metric-chip">
                    <span className="metric-chip-label">Class ID</span>
                    <span className="metric-chip-value">
                      {predictionResult.is_recognized && predictionResult.predicted_class_id !== null && predictionResult.predicted_class_id !== undefined
                        ? `Class ${predictionResult.predicted_class_id}`
                        : "N/A"}
                    </span>
                  </div>

                  {predictionResult.frame_count !== undefined && (
                    <div className="metric-chip">
                      <span className="metric-chip-label">Captured Frames</span>
                      <span className="metric-chip-value">
                        {predictionResult.frame_count} frames
                      </span>
                    </div>
                  )}

                  {predictionResult.confidence !== undefined && (
                    <div className="metric-chip">
                      <span className="metric-chip-label">Confidence</span>
                      <span className="metric-chip-value">
                        {(predictionResult.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}

                  {predictionResult.top_distance !== undefined && (
                    <div className="metric-chip">
                      <span className="metric-chip-label">DTW Distance</span>
                      <span className="metric-chip-value">
                        {predictionResult.top_distance.toFixed(4)}
                      </span>
                    </div>
                  )}

                  {predictionResult.runner_up_distance !== null && predictionResult.runner_up_distance !== undefined && (
                    <div className="metric-chip">
                      <span className="metric-chip-label">Runner-Up Dist</span>
                      <span className="metric-chip-value">
                        {Number(predictionResult.runner_up_distance).toFixed(4)}
                      </span>
                    </div>
                  )}

                  {predictionResult.margin_to_second !== undefined && (
                    <div className="metric-chip">
                      <span className="metric-chip-label">Margin to 2nd</span>
                      <span className="metric-chip-value">
                        +{predictionResult.margin_to_second.toFixed(4)}
                      </span>
                    </div>
                  )}

                  {predictionResult.model_inference_latency_ms !== null && predictionResult.model_inference_latency_ms !== undefined && (
                    <div className="metric-chip">
                      <span className="metric-chip-label">Model Latency</span>
                      <span className="metric-chip-value">
                        {Number(predictionResult.model_inference_latency_ms).toFixed(2)} ms
                      </span>
                    </div>
                  )}

                  {predictionResult.matching_latency_ms !== undefined && (
                    <div className="metric-chip">
                      <span className="metric-chip-label">DTW Latency</span>
                      <span className="metric-chip-value">
                        {predictionResult.matching_latency_ms} ms
                      </span>
                    </div>
                  )}

                  <div className="metric-chip">
                    <span className="metric-chip-label">Full Pipeline Time</span>
                    <span className="metric-chip-value">
                      {predictionResult.processing_time_ms} ms
                    </span>
                  </div>
                </div>

                {/* 5-Class Probabilities (BiGRU Attention) */}
                {predictionResult.class_probabilities && Object.keys(predictionResult.class_probabilities).length > 0 && (
                  <div className="probabilities-card" data-testid="probabilities-card">
                    <div className="probabilities-header">
                      <span>Five-Class Softmax Probabilities</span>
                      <span className="prob-engine-tag">{predictionResult.model_used || "BiGRU Attention"}</span>
                    </div>
                    <div className="probabilities-list">
                      {Object.entries(predictionResult.class_probabilities).map(([sentence, prob], idx) => {
                        const isTop = predictionResult.is_recognized && sentence === predictionResult.predicted_sentence;
                        const pct = (prob * 100).toFixed(1);
                        return (
                          <div key={sentence} className={`prob-row ${isTop ? "prob-top" : ""}`} data-testid={`prob-row-${idx}`}>
                            <div className="prob-label-row">
                              <span className="prob-sentence">
                                <strong>#{idx + 1}</strong> &nbsp; "{sentence}"
                              </span>
                              <span className="prob-value">{pct}%</span>
                            </div>
                            <div className="prob-bar-track">
                              <div
                                className={`prob-bar-fill ${isTop ? "fill-top" : ""}`}
                                style={{ width: `${Math.max(2, Math.min(100, prob * 100))}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* 5-Class Rankings Table (DTW-KNN) */}
                {predictionResult.rankings && predictionResult.rankings.length > 0 && (
                  <div className="rankings-table-wrapper">
                    <div className="rankings-table-title">Full 5-Class Alignment Rankings</div>
                    {predictionResult.rankings.map((rankItem, idx) => {
                      const isTop = idx === 0 && predictionResult.is_recognized;
                      return (
                        <div
                          key={rankItem.class_id}
                          className={`ranking-row ${isTop ? "top-rank" : ""}`}
                        >
                          <span className="ranking-sentence-name">
                            <strong>#{idx + 1}</strong> &nbsp; "{rankItem.sentence}"
                          </span>
                          <div className="ranking-stats">
                            <span>Dist: {rankItem.dtw_distance.toFixed(4)}</span>
                            <span>Sim: {(rankItem.similarity * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                {predictionResult.note && (
                  <p
                    style={{
                      fontSize: "0.78rem",
                      color: "var(--color-text-muted)",
                      margin: "0",
                      fontStyle: "italic",
                    }}
                  >
                    {predictionResult.note}
                  </p>
                )}

                {/* Retry action for webcam */}
                {inputMode === "webcam" && (
                  <div className="retry-action-wrapper">
                    <button
                      type="button"
                      className="btn-secondary-action"
                      onClick={handleRetryRecording}
                      data-testid="btn-retry-webcam"
                    >
                      <RotateCcw size={14} style={{ marginRight: "6px" }} />
                      Record Again / Retry
                    </button>
                  </div>
                )}
              </>
            )}
          </section>
        </div>

        {/* Manual Verification */}
        {predictionResult && (
          <section className="verification-card" aria-label="Manual Verification">
            <div className="card-header-clean">
              <h2 className="card-title-clean">
                <CheckCircle size={18} />
                Manual Verification Against Ground Truth
              </h2>
            </div>

            <div className="verification-form">
              <label htmlFor="actual-sentence-select" style={{ fontWeight: 600, fontSize: "0.9rem" }}>
                Select Actual Signed Sentence:
              </label>
              <select
                id="actual-sentence-select"
                className="verification-select"
                value={actualSentence}
                onChange={(e) => setActualSentence(e.target.value)}
                data-testid="actual-sentence-select"
              >
                <option value="">-- Choose Ground Truth Sentence --</option>
                {demoClasses.map((c) => (
                  <option key={c.demo_class_id} value={c.sentence}>
                    {c.sentence} (Gloss: {c.isl_gloss})
                  </option>
                ))}
                <option value="other_unknown">Out of Vocabulary / Other (Unknown Sentence)</option>
              </select>

              <button
                type="button"
                className="btn-primary-action"
                style={{ flex: "initial", minWidth: "160px" }}
                onClick={handleVerify}
                disabled={!actualSentence}
                data-testid="btn-verify-prediction"
              >
                <Check size={16} />
                Record Verification
              </button>
            </div>

            {/* Verification Result Feedback */}
            {verifiedItem && (
              <div
                className="verification-result-box"
                data-testid="verification-result-box"
              >
                <div>
                  <div style={{ fontSize: "0.9rem", color: "var(--color-text)" }}>
                    <strong>Actual:</strong> "{verifiedItem.actual}" &nbsp;|&nbsp;{" "}
                    <strong>AI Prediction:</strong> "{verifiedItem.predicted}"
                  </div>
                  <div style={{ fontSize: "0.8rem", color: "var(--color-text-muted)", marginTop: "2px" }}>
                    {verifiedItem.isRecognized
                      ? `Model: ${selectedModel === "bigru" ? "BiGRU Attention" : "DTW-KNN"} | Latency: ${verifiedItem.processingTimeMs} ms`
                      : "Input rejected by confidence threshold."}
                  </div>
                </div>
                <span
                  className={`verification-result-badge ${verifiedItem.correctness}`}
                  data-testid="verification-badge"
                >
                  {verifiedItem.correctness === "correct" ? "Correct" : "Incorrect"}
                </span>
              </div>
            )}
          </section>
        )}

        {/* Live Session Statistics */}
        <section aria-label="Live Session Statistics">
          <div className="stats-cards-grid">
            <div className="stat-metric-card" data-testid="stat-total">
              <div className="stat-metric-label">Total Tested</div>
              <div className="stat-metric-value">{sessionStats.total}</div>
            </div>

            <div className="stat-metric-card" data-testid="stat-correct">
              <div className="stat-metric-label">Correct</div>
              <div className="stat-metric-value" style={{ color: "var(--color-status-green)" }}>
                {sessionStats.correct}
              </div>
            </div>

            <div className="stat-metric-card" data-testid="stat-incorrect">
              <div className="stat-metric-label">Incorrect</div>
              <div className="stat-metric-value" style={{ color: "var(--color-status-red)" }}>
                {sessionStats.incorrect}
              </div>
            </div>

            <div className="stat-metric-card" data-testid="stat-rejected">
              <div className="stat-metric-label">Rejected</div>
              <div className="stat-metric-value" style={{ color: "var(--color-status-yellow)" }}>
                {sessionStats.rejected}
              </div>
            </div>

            <div className="stat-metric-card" data-testid="stat-accuracy">
              <div className="stat-metric-label">Manual Accuracy</div>
              <div className="stat-metric-value accuracy">
                {sessionStats.accuracy}%
              </div>
            </div>
          </div>
        </section>

        {/* Testing History Table */}
        <section className="testing-card" aria-label="Session History">
          <div className="card-header-clean">
            <h2 className="card-title-clean">
              <Clock size={18} />
              Testing History ({sessionHistory.length} Entries)
            </h2>
            {sessionHistory.length > 0 && (
              <button
                type="button"
                className="btn-secondary-action"
                onClick={clearHistory}
                style={{ padding: "4px 10px", fontSize: "0.8rem" }}
                data-testid="btn-clear-history"
              >
                Clear History
              </button>
            )}
          </div>

          {sessionHistory.length === 0 ? (
            <p style={{ color: "var(--color-text-muted)", fontSize: "0.9rem", margin: "12px 0" }}>
              No session test records yet. Upload a video or record via webcam, then record a verification above.
            </p>
          ) : (
            <div className="history-table-container">
              <table className="history-data-table" data-testid="history-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Source / Video</th>
                    <th>AI Prediction</th>
                    <th>Actual Sentence</th>
                    <th>Status</th>
                    <th>Result</th>
                    <th>Processing Time</th>
                    <th>Date &amp; Time</th>
                  </tr>
                </thead>
                <tbody>
                  {sessionHistory.map((item, index) => (
                    <tr key={item.id}>
                      <td>{sessionHistory.length - index}</td>
                      <td style={{ fontWeight: 500 }}>{item.filename}</td>
                      <td>"{item.predicted}"</td>
                      <td>"{item.actual}"</td>
                      <td>
                        <span
                          style={{
                            fontSize: "0.8rem",
                            fontWeight: 600,
                            color: item.isRecognized
                              ? "var(--color-status-green)"
                              : "var(--color-status-yellow)",
                          }}
                        >
                          {item.isRecognized ? "Recognized" : "Unknown"}
                        </span>
                      </td>
                      <td>
                        <span className={`badge-res ${item.correctness}`}>
                          {item.correctness === "correct" ? (
                            <>
                              <Check size={12} style={{ display: "inline", marginRight: "3px" }} />
                              Correct
                            </>
                          ) : (
                            <>
                              <X size={12} style={{ display: "inline", marginRight: "3px" }} />
                              Incorrect
                            </>
                          )}
                        </span>
                      </td>
                      <td>{item.processingTimeMs} ms</td>
                      <td style={{ fontSize: "0.82rem", color: "var(--color-text-muted)" }}>
                        {item.dateTime || item.timestamp}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </PageContainer>
  );
}
