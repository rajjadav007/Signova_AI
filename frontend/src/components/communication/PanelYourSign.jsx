import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Volume2,
  VolumeX,
  RotateCcw,
  Square,
  CheckCircle2,
  AlertTriangle,
  Sparkles,
  ArrowRight,
  Loader2,
  X,
  Play,
  Film,
  Upload,
} from "lucide-react";
import LiveWebcamPreview from "../camera/LiveWebcamPreview";
import {
  translateText,
  synthesizeSpeech,
  recognizeSentenceLandmarks,
  recognizeSentenceVideo,
  recognizeFiveClassSentenceLandmarks,
  recognizeFiveClassSentenceVideo,
} from "../../services/api";

export const VOCABULARY_SIGNS = [
  "Food",
  "Hello",
  "Help",
  "No",
  "Please",
  "Sorry",
  "Stop",
  "Thank you",
  "Water",
  "Yes",
];

// Local dictionary for immediate, zero-latency translations of the 10 vocabulary signs
export const STATIC_SIGN_TRANSLATIONS = {
  Food: {
    "en-IN": "Food",
    "hi-IN": "खाना",
    "gu-IN": "ખોરાક",
    "kn-IN": "ಆಹಾರ",
    "ta-IN": "உணவு",
  },
  Hello: {
    "en-IN": "Hello",
    "hi-IN": "नमस्ते",
    "gu-IN": "નમસ્તે",
    "kn-IN": "ನಮಸ್ಕಾರ",
    "ta-IN": "வணக்கம்",
  },
  Help: {
    "en-IN": "Help",
    "hi-IN": "मदद",
    "gu-IN": "મદદ",
    "kn-IN": "ಸಹಾಯ",
    "ta-IN": "உதவி",
  },
  No: {
    "en-IN": "No",
    "hi-IN": "नहीं",
    "gu-IN": "ના",
    "kn-IN": "ಇಲ್ಲ",
    "ta-IN": "இல்லை",
  },
  Please: {
    "en-IN": "Please",
    "hi-IN": "कृपया",
    "gu-IN": "કૃપા કરીને",
    "kn-IN": "ದಯವಿಟ್ಟು",
    "ta-IN": "தயவுசெய்து",
  },
  Sorry: {
    "en-IN": "Sorry",
    "hi-IN": "माफ़ कीजिए",
    "gu-IN": "માફ કરશો",
    "kn-IN": "ಕ್ಷಮಿಸಿ",
    "ta-IN": "மன்னிக்கவும்",
  },
  Stop: {
    "en-IN": "Stop",
    "hi-IN": "रुकिए",
    "gu-IN": "રોકો",
    "kn-IN": "ನಿಲ್ಲಿಸಿ",
    "ta-IN": "நிறுத்து",
  },
  "Thank you": {
    "en-IN": "Thank you",
    "hi-IN": "धन्यवाद",
    "gu-IN": "આભાર",
    "kn-IN": "ಧನ್ಯವಾದ",
    "ta-IN": "நன்றி",
  },
  Water: {
    "en-IN": "Water",
    "hi-IN": "पानी",
    "gu-IN": "પાણી",
    "kn-IN": "ನೀರು",
    "ta-IN": "தண்ணீர்",
  },
  Yes: {
    "en-IN": "Yes",
    "hi-IN": "हाँ",
    "gu-IN": "હા",
    "kn-IN": "ಹೌದು",
    "ta-IN": "ஆம்",
  },
};

// In-memory audio base64 cache key: `${word}_${targetLanguage}`
const audioCache = new Map();

export default function PanelYourSign({
  otherLanguage = "en-IN",
  otherLanguageName = "English",
  recognitionMode = "isolated",
  onModeChange = null,
  sentenceModel = "bigru",
  onSentenceModelChange = null,
}) {
  const [currentSign, setCurrentSign] = useState(null);
  const [translatedText, setTranslatedText] = useState("");
  const [isTranslating, setIsTranslating] = useState(false);
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [audioBlocked, setAudioBlocked] = useState(false);
  const [uncertainSign, setUncertainSign] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // Sentence recognition states
  const [sentenceResult, setSentenceResult] = useState(null);
  const [isSentenceInferring, setIsSentenceInferring] = useState(false);
  const [sentenceError, setSentenceError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);

  const fileInputRef = useRef(null);
  const webcamRef = useRef(null);
  const audioRef = useRef(null);
  const lastProcessedSignRef = useRef({ sign: null, timestamp: 0 });
  const activeLangRef = useRef(otherLanguage);
  const latestReqIdRef = useRef(0);

  // Online / offline network detection
  useEffect(() => {
    const handleOnline = () => {
      if (errorMessage && errorMessage.includes("offline")) {
        setErrorMessage(null);
      }
    };
    const handleOffline = () => {
      setErrorMessage("Network disconnected. You are currently offline. Check your internet connection.");
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, [errorMessage]);

  useEffect(() => {
    activeLangRef.current = otherLanguage;
    // Re-translate current sign if language selection changed
    if (currentSign) {
      handleTranslateAndSpeak(currentSign, otherLanguage);
    }
  }, [otherLanguage]);

  useEffect(() => {
    return () => {
      if (audioRef.current && typeof audioRef.current.pause === "function") {
        try {
          audioRef.current.pause();
        } catch {
          // ignore
        }
        audioRef.current = null;
      }
    };
  }, []);

  const stopAudio = useCallback(() => {
    if (audioRef.current && typeof audioRef.current.pause === "function") {
      try {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      } catch {
        // ignore
      }
    }
    setIsPlayingAudio(false);
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // ignore
      }
    }
  }, []);

  const playBase64Audio = useCallback((base64Wav) => {
    try {
      stopAudio();
      const audioUrl = `data:audio/wav;base64,${base64Wav}`;
      const audio = new Audio(audioUrl);
      audioRef.current = audio;

      audio.onended = () => {
        setIsPlayingAudio(false);
      };
      audio.onerror = () => {
        setIsPlayingAudio(false);
      };

      setIsPlayingAudio(true);
      setAudioBlocked(false);

      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.catch((err) => {
          setIsPlayingAudio(false);
          if (err && err.name === "NotAllowedError") {
            setAudioBlocked(true);
            setErrorMessage("Audio playback was blocked by browser autoplay policy. Click 'Play Audio' to listen.");
          }
        });
      }
    } catch {
      setIsPlayingAudio(false);
    }
  }, [stopAudio]);

  const speakWithBrowserFallback = useCallback((text, langCode) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = langCode || "en-US";
      utterance.rate = 1.0;
      utterance.onend = () => setIsPlayingAudio(false);
      utterance.onerror = () => setIsPlayingAudio(false);
      setIsPlayingAudio(true);
      window.speechSynthesis.speak(utterance);
    } catch {
      setIsPlayingAudio(false);
    }
  }, []);

  const handleTranslateAndSpeak = useCallback(
    async (signName, targetLang) => {
      if (!signName || signName.toLowerCase() === "unknown" || signName.toLowerCase() === "uncertain") {
        return;
      }
      const cleanSign =
        VOCABULARY_SIGNS.find(
          (v) => v.toLowerCase() === signName.trim().toLowerCase()
        ) || signName.trim();

      const reqId = ++latestReqIdRef.current;

      setCurrentSign(cleanSign);
      setErrorMessage(null);
      setAudioBlocked(false);

      // Check static dictionary first
      let translated = "";
      if (STATIC_SIGN_TRANSLATIONS[cleanSign]?.[targetLang]) {
        translated = STATIC_SIGN_TRANSLATIONS[cleanSign][targetLang];
      }

      // If not in static dictionary, fetch from Sarvam translate API
      if (!translated) {
        setIsTranslating(true);
        try {
          const res = await translateText(cleanSign, targetLang, "en-IN");
          if (reqId !== latestReqIdRef.current) return;
          translated = res.translated_text || cleanSign;
        } catch (err) {
          if (reqId !== latestReqIdRef.current) return;
          translated = cleanSign;
          setErrorMessage(err.message || "Translation service unavailable; using original label.");
        } finally {
          if (reqId === latestReqIdRef.current) {
            setIsTranslating(false);
          }
        }
      }

      if (reqId !== latestReqIdRef.current) return;
      setTranslatedText(translated);

      // Generate or retrieve audio from cache
      const cacheKey = `${translated}_${targetLang}`;
      if (audioCache.has(cacheKey)) {
        playBase64Audio(audioCache.get(cacheKey));
        return;
      }

      setIsSynthesizing(true);
      try {
        const ttsRes = await synthesizeSpeech(translated, targetLang, "priya");
        if (reqId !== latestReqIdRef.current) return;
        if (ttsRes && ttsRes.audio_base64) {
          audioCache.set(cacheKey, ttsRes.audio_base64);
          playBase64Audio(ttsRes.audio_base64);
        } else {
          speakWithBrowserFallback(translated, targetLang);
        }
      } catch (err) {
        if (reqId !== latestReqIdRef.current) return;
        // Fallback gracefully to browser voice without throwing
        speakWithBrowserFallback(translated, targetLang);
      } finally {
        if (reqId === latestReqIdRef.current) {
          setIsSynthesizing(false);
        }
      }
    },
    [playBase64Audio, speakWithBrowserFallback]
  );

  // Callback from LiveWebcamPreview on confirmed prediction (>= 0.80)
  const handleSignRecognized = useCallback(
    (signName) => {
      if (!signName) return;
      const now = Date.now();
      const clean = signName.trim().toLowerCase();

      // Duplicate prevention: ignore identical sign within 2.5s to avoid repeated speech on held gestures
      if (
        lastProcessedSignRef.current.sign === clean &&
        now - lastProcessedSignRef.current.timestamp < 2500
      ) {
        return;
      }

      lastProcessedSignRef.current = { sign: clean, timestamp: now };
      setUncertainSign(null);
      handleTranslateAndSpeak(signName, activeLangRef.current);
    },
    [handleTranslateAndSpeak]
  );

  // Callback from LiveWebcamPreview on low-confidence suggestion (< 0.80)
  const handleUncertainSuggestion = useCallback((signName) => {
    if (!signName) return;
    const clean =
      VOCABULARY_SIGNS.find(
        (v) => v.toLowerCase() === signName.trim().toLowerCase()
      ) || signName.trim();
    // Do NOT auto-speak uncertain signs; set suggestion for user confirmation
    setUncertainSign(clean);
  }, []);

  const confirmUncertainSign = (confirmedSign) => {
    setUncertainSign(null);
    lastProcessedSignRef.current = {
      sign: confirmedSign.toLowerCase(),
      timestamp: Date.now(),
    };
    handleTranslateAndSpeak(confirmedSign, activeLangRef.current);
  };

  const dismissUncertainSign = () => {
    setUncertainSign(null);
  };

  const clearResult = () => {
    stopAudio();
    setCurrentSign(null);
    setSentenceResult(null);
    setTranslatedText("");
    setUncertainSign(null);
    setErrorMessage(null);
    setSentenceError(null);
    setAudioBlocked(false);
  };

  useEffect(() => {
    stopAudio();
    setErrorMessage(null);
    setSentenceError(null);
  }, [recognitionMode, stopAudio]);

  const handleSentenceSuccess = useCallback(
    async (response) => {
      if (!response) return;

      // Handle insufficient frames error separately
      if (response.status === "INSUFFICIENT_FRAMES") {
        setSentenceResult(null);
        setSentenceError(
          response.reason || "Insufficient frames collected. Please perform the full sign."
        );
        return;
      }

      // Handle no valid landmarks error separately
      if (response.status === "NO_VALID_LANDMARKS") {
        setSentenceResult(null);
        setSentenceError(
          response.reason || "No valid hand/pose landmarks detected in video or camera feed."
        );
        return;
      }

      // Handle general error / failure with no prediction
      if (response.status === "ERROR" || (response.status === "failed" && !response.predicted_sentence && !response.sentence)) {
        setSentenceResult(null);
        setSentenceError(response.reason || "Sentence recognition failed.");
        return;
      }

      // Valid inference result exists - always display top predicted sentence and probabilities
      setSentenceResult(response);
      setSentenceError(null);

      const sentenceText =
        response.predicted_sentence ||
        response.sentence ||
        response.predicted_text ||
        response.text ||
        response.top_predictions?.[0]?.sentence ||
        "";

      const isConfirmed = response.is_recognized === true;

      // Only treat high confidence predictions as confirmed translation & auto-speak
      if (sentenceText) {
        if (isConfirmed) {
          await handleTranslateAndSpeak(sentenceText, activeLangRef.current);
        } else {
          // For low-confidence, fetch translation text for display but do NOT auto-speak audio
          try {
            const res = await translateText(sentenceText, activeLangRef.current, "en-IN");
            setTranslatedText(res?.translated_text || sentenceText);
          } catch {
            setTranslatedText(sentenceText);
          }
        }
      }
    },
    [handleTranslateAndSpeak]
  );

  const handleRecordSentence = useCallback(async () => {
    if (isSentenceInferring) return;

    setSentenceError(null);
    setIsSentenceInferring(true);

    try {
      let frames = [];
      if (webcamRef.current && typeof webcamRef.current.getFrames === "function") {
        frames = webcamRef.current.getFrames() || [];
      }

      let timestamps = [];
      let landmarks = [];

      if (frames && frames.length > 0) {
        timestamps = frames.map((f, i) => Math.round(f.timestamp || i * 40));
        landmarks = frames.map((f) => {
          if (f.landmarks && Array.isArray(f.landmarks)) return f.landmarks;
          return new Array(300).fill(0);
        });
      } else {
        timestamps = Array.from({ length: 30 }, (_, i) => i * 40);
        landmarks = Array.from({ length: 30 }, () => new Array(300).fill(0));
      }

      const response = await recognizeFiveClassSentenceLandmarks(landmarks, sentenceModel, 0.20, 10);
      await handleSentenceSuccess(response);
    } catch (err) {
      setSentenceError(err.message || "Continuous sentence recognition failed. Please try again.");
    } finally {
      setIsSentenceInferring(false);
    }
  }, [isSentenceInferring, handleSentenceSuccess, sentenceModel]);

  const handleFileUpload = useCallback(
    async (event) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const validTypes = [
        "video/mp4",
        "video/webm",
        "video/quicktime",
        "video/x-matroska",
        "video/avi",
      ];
      const isVideoExt = /\.(mp4|webm|mov|avi|mkv)$/i.test(file.name);
      if (!validTypes.includes(file.type) && !isVideoExt) {
        setSentenceError("Please upload a valid .mp4 or .webm video file.");
        return;
      }

      setSentenceError(null);
      setIsSentenceInferring(true);
      setUploadProgress(0);

      try {
        const response = await recognizeFiveClassSentenceVideo(
          file,
          sentenceModel,
          (percent) => {
            setUploadProgress(percent);
          }
        );
        await handleSentenceSuccess(response);
      } catch (err) {
        setSentenceError(err.message || "Failed to recognize sentence from uploaded video.");
      } finally {
        setIsSentenceInferring(false);
        setUploadProgress(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    },
    [handleSentenceSuccess, sentenceModel]
  );

  const replayAudio = () => {
    if (!translatedText) return;
    const cacheKey = `${translatedText}_${otherLanguage}`;
    if (audioCache.has(cacheKey)) {
      playBase64Audio(audioCache.get(cacheKey));
    } else {
      handleTranslateAndSpeak(currentSign || translatedText, otherLanguage);
    }
  };

  // Derived continuous sentence display values
  const recognizedSentence =
    sentenceResult?.sentence ||
    sentenceResult?.predicted_sentence ||
    sentenceResult?.predicted_text ||
    sentenceResult?.text ||
    sentenceResult?.predicted_sign ||
    sentenceResult?.top_predictions?.[0]?.sentence ||
    sentenceResult?.top_3_predictions?.[0]?.sentence ||
    currentSign ||
    "";

  const rawConfidence =
    typeof sentenceResult?.confidence === "number"
      ? sentenceResult.confidence
      : typeof sentenceResult?.top_predictions?.[0]?.confidence === "number"
      ? sentenceResult.top_predictions[0].confidence
      : typeof sentenceResult?.top_3_predictions?.[0]?.confidence === "number"
      ? sentenceResult.top_3_predictions[0].confidence
      : 0;

  const confPercent = rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence;
  const formattedConfidence = Number.isInteger(Number(confPercent.toFixed(1)))
    ? `${Math.round(confPercent)}%`
    : `${confPercent.toFixed(1)}%`;

  const glossTokens = Array.isArray(sentenceResult?.gloss_tokens)
    ? sentenceResult.gloss_tokens
    : Array.isArray(sentenceResult?.tokens)
    ? sentenceResult.tokens
    : Array.isArray(sentenceResult?.glosses)
    ? sentenceResult.glosses
    : typeof sentenceResult?.isl_gloss === "string"
    ? sentenceResult.isl_gloss.split(/\s+/).filter(Boolean)
    : typeof sentenceResult?.gloss === "string"
    ? sentenceResult.gloss.split(/\s+/).filter(Boolean)
    : [];

  const rawAlternatives = Array.isArray(sentenceResult?.top_predictions)
    ? sentenceResult.top_predictions
    : Array.isArray(sentenceResult?.top_3_predictions)
    ? sentenceResult.top_3_predictions
    : Array.isArray(sentenceResult?.alternatives)
    ? sentenceResult.alternatives
    : [];
  const topAlternatives = rawAlternatives.slice(0, 3);

  return (
    <div
      className="communication-panel panel-your-sign"
      data-testid="panel-your-sign"
      aria-label="Direction A: Your Sign to Spoken Audio"
    >
      <header className="panel-header">
        <div className="panel-title-group">
          <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
            <span className="panel-direction-badge deaf-badge">Deaf Person → Hearing Person</span>
            {recognitionMode === "sentence" && (
              <span className="sentence-mode-badge" data-testid="sentence-mode-badge">
                Continuous Sentence Recognition (5 ISL Sentences &bull; {sentenceModel === "bigru" ? "BiGRU Attention" : "DTW-KNN"})
              </span>
            )}
          </div>
          <h2 className="panel-title">Your Sign</h2>
        </div>
        <span className="panel-target-lang-pill" data-testid="panel-a-target-lang">
          Speaking: <strong>{otherLanguageName}</strong>
        </span>
      </header>

      {/* Webcam Feed (Single Camera Instance, initialAutoSpeak false to route through Sarvam TTS) */}
      <div className="panel-camera-section">
        <LiveWebcamPreview
          ref={webcamRef}
          recognitionMode={recognitionMode}
          sentenceModel={sentenceModel}
          onSignRecognized={handleSignRecognized}
          onUncertainSuggestion={handleUncertainSuggestion}
          onSentenceRecognized={handleSentenceSuccess}
          initialAutoSpeak={false}
          showSpeechToggle={false}
        />
      </div>

      {/* Continuous Sentence Mode Actions Bar */}
      {recognitionMode === "sentence" && (
        <div className="sentence-actions-bar" data-testid="sentence-actions-bar">
          <button
            type="button"
            className="sentence-action-btn record-sentence-btn"
            data-testid="record-sentence-btn"
            onClick={handleRecordSentence}
            disabled={isSentenceInferring}
            title="Capture sequence from webcam buffer and recognize sentence"
          >
            {isSentenceInferring ? (
              <>
                <Loader2 size={16} className="rotating" />
                <span>Recognizing...</span>
              </>
            ) : (
              <>
                <Film size={16} />
                <span>Record Sentence</span>
              </>
            )}
          </button>

          <input
            type="file"
            ref={fileInputRef}
            accept="video/mp4,video/webm"
            data-testid="sentence-file-input"
            onChange={handleFileUpload}
            style={{ display: "none" }}
          />
          <button
            type="button"
            className="sentence-action-btn upload-video-btn"
            data-testid="sentence-upload-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={isSentenceInferring}
            title="Upload MP4 or WebM video file for sentence recognition"
          >
            <Upload size={16} />
            <span>Upload Video (.mp4 / .webm)</span>
          </button>
        </div>
      )}

      {/* Upload Progress Bar */}
      {recognitionMode === "sentence" && uploadProgress !== null && (
        <div className="upload-progress-container" data-testid="upload-progress-bar">
          <div className="upload-progress-bar">
            <div className="upload-progress-fill" style={{ width: `${uploadProgress}%` }} />
          </div>
          <span className="upload-progress-label">Uploading video: {uploadProgress}%</span>
        </div>
      )}

      {/* Inference Loading Spinner */}
      {recognitionMode === "sentence" && isSentenceInferring && (
        <div className="sentence-loading-card" data-testid="sentence-loading-spinner" role="status">
          <Loader2 size={24} className="rotating" />
          <div className="sentence-loading-text">
            <strong>Recognizing Continuous Sentence...</strong>
            <p>Analyzing temporal sequence across 5 ISL Sentences with {sentenceModel === "bigru" ? "BiGRU Attention" : "DTW-KNN"} model</p>
          </div>
        </div>
      )}

      {/* Sentence Error Display */}
      {recognitionMode === "sentence" && sentenceError && (
        <div className="panel-error-notice" role="alert" data-testid="sentence-error-notice">
          <AlertTriangle size={16} />
          <span>{sentenceError}</span>
        </div>
      )}

      {/* Continuous Sentence Recognition Result Card */}
      {recognitionMode === "sentence" && (sentenceResult || (currentSign && !isSentenceInferring)) && (
        <section
          className="sentence-communication-result-card"
          data-testid="sentence-communication-result-card"
          aria-label="Recognized ISL Sentence and Spoken Translation"
        >
          {/* Prominent English Sentence & Confidence Score */}
          <div className="sentence-result-main-header">
            <div className="sentence-text-group">
              <span
                className={`result-badge-label ${sentenceResult?.is_recognized !== false ? "confirmed-label" : "low-confidence-label"}`}
                data-testid="sentence-prediction-type"
              >
                {sentenceResult?.is_recognized !== false ? "Prediction" : "Low confidence"}
              </span>
              <h3 className="recognized-sentence-text" data-testid="recognized-sentence-text">
                {recognizedSentence}
              </h3>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "4px" }}>
              {rawConfidence > 0 && (
                <span
                  className={`sentence-confidence-badge ${sentenceResult?.is_recognized !== false ? "" : "low-confidence"}`}
                  data-testid="sentence-confidence-score"
                >
                  {formattedConfidence}
                </span>
              )}
              {sentenceResult?.margin_to_second !== undefined && sentenceResult?.margin_to_second !== null && (
                <span className="sentence-margin-chip" data-testid="sentence-margin-chip">
                  +{((sentenceResult.margin_to_second || 0) * 100).toFixed(1)}% vs 2nd
                </span>
              )}
            </div>
          </div>

          {/* Low Confidence Warning Notice */}
          {sentenceResult?.is_recognized === false && (
            <div className="sentence-low-confidence-banner" data-testid="sentence-low-confidence-banner" role="alert">
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <AlertTriangle size={16} />
                <span>Low confidence prediction. Not automatically confirmed or spoken.</span>
              </div>
              <button
                type="button"
                className="tentative-speak-btn"
                data-testid="tentative-speak-btn"
                onClick={() => handleTranslateAndSpeak(recognizedSentence, activeLangRef.current)}
              >
                Translate &amp; Speak
              </button>
            </div>
          )}

          {/* Five-Class Probability Distribution */}
          {sentenceResult?.class_probabilities && Object.keys(sentenceResult.class_probabilities).length > 0 && (
            <div className="five-class-probabilities" data-testid="five-class-probabilities">
              <div className="prob-section-title">Class Probabilities (5 ISL Sentences):</div>
              <div className="prob-bar-grid">
                {Object.entries(sentenceResult.class_probabilities).map(([sentence, prob]) => {
                  const probVal = typeof prob === "number" ? prob : 0;
                  const probPct = Math.min(100, Math.max(0, probVal * 100));
                  const isTop =
                    sentence.toLowerCase() === recognizedSentence.toLowerCase() ||
                    sentence.toLowerCase() === (sentenceResult.predicted_sentence || "").toLowerCase();
                  return (
                    <div
                      key={sentence}
                      className={`prob-row ${isTop ? "top-prob" : ""}`}
                      data-testid={`prob-row-${sentence.replace(/[\s.]+/g, "-")}`}
                    >
                      <span className="prob-class-name" title={sentence}>
                        {sentence}
                      </span>
                      <div className="prob-track">
                        <div
                          className="prob-fill"
                          style={{ width: `${probPct.toFixed(1)}%` }}
                        />
                      </div>
                      <span className="prob-percent">{probPct.toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Gloss Tokens */}
          {glossTokens.length > 0 && (
            <div className="sentence-gloss-section">
              <span className="sentence-sublabel">ISL Gloss Sequence:</span>
              <div className="sentence-gloss-tokens" data-testid="sentence-gloss-tokens">
                {glossTokens.map((token, idx) => (
                  <span key={`gloss-${token}-${idx}`} className="gloss-token-chip">
                    {token}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Top-3 Alternative Sentences */}
          {topAlternatives.length > 0 && (
            <div className="sentence-alternatives-section">
              <span className="sentence-sublabel">Alternative Candidates (Top 3):</span>
              <div className="sentence-top-alternatives" data-testid="sentence-top-alternatives">
                {topAlternatives.map((alt, idx) => {
                  const altText =
                    alt.sentence || alt.text || alt.predicted_sentence || alt.label || alt.sign || "";
                  const altConf = typeof alt.confidence === "number" ? alt.confidence : 0;
                  const altConfPct = altConf <= 1 ? Math.round(altConf * 100) : Math.round(altConf);
                  const altGloss =
                    alt.gloss ||
                    alt.isl_gloss ||
                    (Array.isArray(alt.gloss_tokens) ? alt.gloss_tokens.join(" ") : "");
                  return (
                    <button
                      key={`alt-${idx}`}
                      type="button"
                      className="alt-sentence-item"
                      data-testid={`alt-candidate-${idx}`}
                      onClick={() => handleTranslateAndSpeak(altText, otherLanguage)}
                      title="Translate and speak this candidate sentence"
                    >
                      <span className="alt-rank">#{idx + 1}</span>
                      <div className="alt-info">
                        <span className="alt-text">{altText}</span>
                        {altGloss && <span className="alt-gloss">Gloss: {altGloss}</span>}
                      </div>
                      {altConf > 0 && <span className="alt-conf">{altConfPct}%</span>}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Spoken Translation in Other Language */}
          <div className="sentence-translation-wrap">
            <span className="result-badge-label">Spoken in {otherLanguageName}</span>
            <div className="translated-sign-text" data-testid="translated-sign-text">
              {isTranslating ? (
                <span className="translating-indicator">
                  <Loader2 size={16} className="rotating" /> Translating...
                </span>
              ) : (
                translatedText || recognizedSentence
              )}
            </div>
          </div>

          {/* Audio Playback Controls */}
          <div className="result-playback-bar">
            <div className="audio-status-indicator" data-testid="audio-status-indicator">
              {isPlayingAudio ? (
                <span className="audio-playing-pill">
                  <Volume2 size={16} className="pulse-icon" />
                  Playing spoken audio ({otherLanguageName})...
                </span>
              ) : isSynthesizing ? (
                <span className="audio-synthesizing-pill">
                  <Loader2 size={16} className="rotating" />
                  Generating speech...
                </span>
              ) : (
                <span className="audio-ready-pill">
                  <CheckCircle2 size={16} />
                  Spoken in {otherLanguageName}
                </span>
              )}
            </div>

            <div className="playback-buttons-group">
              {isPlayingAudio ? (
                <button
                  type="button"
                  className="playback-btn stop-audio-btn"
                  data-testid="stop-audio-btn"
                  onClick={stopAudio}
                  aria-label="Stop audio"
                  title="Stop audio playback"
                >
                  <Square size={14} />
                  <span>Stop</span>
                </button>
              ) : (
                <button
                  type="button"
                  className="playback-btn replay-btn"
                  data-testid="replay-audio-btn"
                  onClick={replayAudio}
                  disabled={!translatedText || isTranslating || isSynthesizing}
                  aria-label="Replay spoken audio"
                  title="Replay spoken translation"
                >
                  <RotateCcw size={14} />
                  <span>Replay</span>
                </button>
              )}

              {audioBlocked && (
                <button
                  type="button"
                  className="playback-btn play-unblock-btn"
                  data-testid="unblock-audio-btn"
                  onClick={replayAudio}
                  aria-label="Play audio"
                >
                  <Play size={14} />
                  <span>Play Audio</span>
                </button>
              )}

              <button
                type="button"
                className="clear-result-btn"
                data-testid="clear-sign-result-btn"
                onClick={clearResult}
                aria-label="Clear current result"
                title="Clear current sentence result"
              >
                Clear
              </button>
            </div>
          </div>

          {/* Model & Latency Telemetry Footer */}
          <div className="sentence-meta-footer" data-testid="sentence-meta-footer">
            <span>
              Model: <strong>{sentenceResult?.model_used || (sentenceModel === "bigru" ? "FiveClassBiGRUAttentionModel" : "FiveClassDTWKNN")}</strong>
            </span>
            {sentenceResult?.processing_time_ms !== undefined && (
              <span>
                Processing: <strong>{Math.round(sentenceResult.processing_time_ms)} ms</strong>
              </span>
            )}
            {sentenceResult?.frame_count !== undefined && (
              <span>
                Frames: <strong>{sentenceResult.frame_count}</strong>
              </span>
            )}
          </div>

          {errorMessage && (
            <div className="panel-error-notice" role="alert">
              <span>{errorMessage}</span>
            </div>
          )}
        </section>
      )}

      {/* ========================================================================= */}
      {/* ISOLATED RECOGNITION MODE (100% UNCHANGED)                               */}
      {/* ========================================================================= */}

      {/* Uncertain Prediction Card (interactive confirm, correct, dismiss) */}
      {recognitionMode === "isolated" && uncertainSign && (
        <div
          className="uncertain-suggestion-card"
          data-testid="uncertain-suggestion-card"
          role="alert"
        >
          <div className="uncertain-card-header">
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <AlertTriangle size={18} className="uncertain-icon" />
              <strong>Uncertain gesture detected</strong>
            </div>
            <button
              type="button"
              className="uncertain-dismiss-btn"
              onClick={dismissUncertainSign}
              aria-label="Dismiss suggestion"
            >
              <X size={16} />
            </button>
          </div>
          <p className="uncertain-prompt-text">
            Did you mean <strong>"{uncertainSign}"</strong>?
          </p>
          <div className="uncertain-actions-row">
            <button
              type="button"
              className="confirm-sign-btn"
              data-testid="confirm-uncertain-btn"
              onClick={() => confirmUncertainSign(uncertainSign)}
            >
              <CheckCircle2 size={16} />
              <span>Confirm & Speak</span>
            </button>
            <button
              type="button"
              className="dismiss-sign-btn"
              data-testid="dismiss-uncertain-btn"
              onClick={dismissUncertainSign}
            >
              Dismiss
            </button>
          </div>

          {/* Quick-pick vocabulary correction chips */}
          <div className="vocabulary-quick-picks">
            <span className="quick-picks-label">Or choose correct sign:</span>
            <div className="quick-picks-chips">
              {VOCABULARY_SIGNS.map((sign) => (
                <button
                  key={`pick-${sign}`}
                  type="button"
                  className="quick-pick-chip"
                  onClick={() => confirmUncertainSign(sign)}
                >
                  {sign}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Direction A Result Display Card */}
      {recognitionMode === "isolated" && currentSign && (
        <section
          className="sign-communication-result-card"
          data-testid="sign-communication-result-card"
          aria-label="Recognized Sign and Spoken Translation"
        >
          <div className="result-card-top">
            <div className="sign-badge-wrap">
              <span className="result-badge-label">Recognized ISL Sign</span>
              <span className="recognized-sign-gloss" data-testid="confirmed-sign-gloss">
                {currentSign.toUpperCase()}
              </span>
            </div>

            <ArrowRight size={20} className="translation-arrow" aria-hidden="true" />

            <div className="translation-badge-wrap">
              <span className="result-badge-label">
                Spoken in {otherLanguageName}
              </span>
              <span
                className="translated-sign-text"
                data-testid="translated-sign-text"
              >
                {isTranslating ? (
                  <span className="translating-indicator">
                    <Loader2 size={16} className="rotating" /> Translating...
                  </span>
                ) : (
                  translatedText
                )}
              </span>
            </div>
          </div>

          {/* Audio Playback Controls */}
          <div className="result-playback-bar">
            <div className="audio-status-indicator" data-testid="audio-status-indicator">
              {isPlayingAudio ? (
                <span className="audio-playing-pill">
                  <Volume2 size={16} className="pulse-icon" />
                  Playing spoken audio ({otherLanguageName})...
                </span>
              ) : isSynthesizing ? (
                <span className="audio-synthesizing-pill">
                  <Loader2 size={16} className="rotating" />
                  Generating speech...
                </span>
              ) : (
                <span className="audio-ready-pill">
                  <CheckCircle2 size={16} />
                  Spoken in {otherLanguageName}
                </span>
              )}
            </div>

            <div className="playback-buttons-group">
              {isPlayingAudio ? (
                <button
                  type="button"
                  className="playback-btn stop-audio-btn"
                  data-testid="stop-audio-btn"
                  onClick={stopAudio}
                  aria-label="Stop audio"
                  title="Stop audio playback"
                >
                  <Square size={14} />
                  <span>Stop</span>
                </button>
              ) : (
                <button
                  type="button"
                  className="playback-btn replay-btn"
                  data-testid="replay-audio-btn"
                  onClick={replayAudio}
                  disabled={!translatedText || isTranslating || isSynthesizing}
                  aria-label="Replay spoken audio"
                  title="Replay spoken translation"
                >
                  <RotateCcw size={14} />
                  <span>Replay</span>
                </button>
              )}

              {audioBlocked && (
                <button
                  type="button"
                  className="playback-btn play-unblock-btn"
                  data-testid="unblock-audio-btn"
                  onClick={replayAudio}
                  aria-label="Play audio"
                >
                  <Play size={14} />
                  <span>Play Audio</span>
                </button>
              )}

              <button
                type="button"
                className="clear-result-btn"
                data-testid="clear-sign-result-btn"
                onClick={clearResult}
                aria-label="Clear current result"
                title="Clear current sign result"
              >
                Clear
              </button>
            </div>
          </div>

          {errorMessage && (
            <div className="panel-error-notice" role="alert" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "10px" }}>
              <span>{errorMessage}</span>
              <button
                type="button"
                className="panel-retry-btn"
                data-testid="panel-a-retry-btn"
                onClick={() => currentSign && handleTranslateAndSpeak(currentSign, otherLanguage)}
                aria-label="Retry translation"
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "4px",
                  padding: "3px 8px",
                  borderRadius: "var(--radius-sm)",
                  border: "1px solid rgba(239, 68, 68, 0.4)",
                  backgroundColor: "rgba(239, 68, 68, 0.15)",
                  color: "#DC2626",
                  cursor: "pointer",
                  fontSize: "0.78rem",
                  fontWeight: "600",
                  flexShrink: 0,
                }}
              >
                <RotateCcw size={12} />
                <span>Retry</span>
              </button>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
