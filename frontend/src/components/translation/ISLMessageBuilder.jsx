import React, { useEffect, useRef, useState } from "react";
import {
  Volume2,
  VolumeX,
  Play,
  RotateCcw,
  Trash2,
  Languages,
  CheckCircle2,
  AlertCircle,
  Video,
  VideoOff,
  PlusCircle,
  X,
  Loader2,
  KeyRound,
} from "lucide-react";
import {
  recognizeIsolatedVideo,
  translateText,
  synthesizeSpeech,
  fetchSupportedLanguages,
} from "../../services/api";

export const SUPPORTED_SIGNS = [
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

export const DEFAULT_LANGUAGES = {
  "hi-IN": "Hindi (हिन्दी)",
  "en-IN": "English (Indian)",
  "gu-IN": "Gujarati (ગુજરાતી)",
  "kn-IN": "Kannada (ಕನ್ನಡ)",
  "ta-IN": "Tamil (தமிழ்)",
  "bn-IN": "Bengali (বাংলা)",
  "te-IN": "Telugu (తెలుగు)",
  "ml-IN": "Malayalam (മലയാളം)",
  "mr-IN": "Marathi (मराठी)",
  "pa-IN": "Punjabi (ਪੰਜਾਬੀ)",
  "od-IN": "Odia (ଓଡ଼ିଆ)",
};

export const SUPPORTED_VOICES = [
  { id: "meera", label: "Meera (Female, Warm)" },
  { id: "arvind", label: "Arvind (Male, Professional)" },
  { id: "pavithra", label: "Pavithra (Female, Expressive)" },
];

export default function ISLMessageBuilder({
  hideCameraPreview = false,
  externalRecognizedSign = null,
  externalUncertainSuggestion = null,
} = {}) {
  // --- Message Builder State ---
  const [words, setWords] = useState([]);
  const [messageText, setMessageText] = useState("");
  const [uncertainSuggestion, setUncertainSuggestion] = useState(null);

  // --- Recognition State ---
  const [cameraActive, setCameraActive] = useState(false);
  const [recording, setRecording] = useState(false);
  const [countdown, setCountdown] = useState(3);
  const [recognitionLoading, setRecognitionLoading] = useState(false);
  const [recognitionResult, setRecognitionResult] = useState(null);
  const [cameraError, setCameraError] = useState(null);

  // --- Translation State ---
  const [targetLanguage, setTargetLanguage] = useState("hi-IN");
  const [availableLanguages, setAvailableLanguages] = useState(DEFAULT_LANGUAGES);
  const [translating, setTranslating] = useState(false);
  const [translatedText, setTranslatedText] = useState("");
  const [translationError, setTranslationError] = useState(null);

  // --- Text-to-Speech State ---
  const [selectedVoice, setSelectedVoice] = useState("meera");
  const [synthesizing, setSynthesizing] = useState(false);
  const [audioBase64, setAudioBase64] = useState(null);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [speechError, setSpeechError] = useState(null);

  // --- Refs ---
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const countdownIntervalRef = useRef(null);
  const audioInstanceRef = useRef(null);
  const lastRecognizedTimestampRef = useRef(0);

  // Sync external recognition events (from LiveWebcamPreview)
  useEffect(() => {
    if (externalRecognizedSign && externalRecognizedSign.word) {
      handleAddWord(externalRecognizedSign.word, externalRecognizedSign.timestamp || Date.now());
    }
  }, [externalRecognizedSign]);

  // Sync external uncertain suggestions (from LiveWebcamPreview)
  useEffect(() => {
    if (externalUncertainSuggestion && externalUncertainSuggestion.word) {
      setUncertainSuggestion(externalUncertainSuggestion.word);
    }
  }, [externalUncertainSuggestion]);

  // Load language catalog on mount
  useEffect(() => {
    let mounted = true;
    try {
      const p = fetchSupportedLanguages();
      if (p && typeof p.then === "function") {
        p.then((langs) => {
          if (mounted && langs && Object.keys(langs).length > 0) {
            setAvailableLanguages((prev) => ({ ...prev, ...langs }));
          }
        }).catch(() => {
          // Fallback to DEFAULT_LANGUAGES
        });
      }
    } catch {
      // ignore
    }

    return () => {
      mounted = false;
      if (!hideCameraPreview) {
        stopCamera();
      }
      if (audioInstanceRef.current) {
        audioInstanceRef.current.pause();
        audioInstanceRef.current = null;
      }
    };
  }, [hideCameraPreview]);

  // Sync words array changes to editable messageText
  const updateWordsAndText = (newWords) => {
    setWords(newWords);
    setMessageText(newWords.join(" "));
  };

  // Add recognized word to sequence (preventing duplicate events within 1.5s)
  const handleAddWord = (signName, eventId = null) => {
    if (!signName || typeof signName !== "string") return;
    const cleanWord = signName.trim();
    if (!cleanWord) return;

    // Duplicate event guard
    const now = Date.now();
    if (eventId && eventId === lastRecognizedTimestampRef.current) {
      return;
    }
    if (eventId) {
      lastRecognizedTimestampRef.current = eventId;
    }

    setWords((prevWords) => {
      const nextWords = [...prevWords, cleanWord];
      setMessageText((prevText) => {
        const trimmed = prevText.trim();
        return trimmed ? `${trimmed} ${cleanWord}` : cleanWord;
      });
      return nextWords;
    });
  };

  // Remove individual word by index
  const handleRemoveWord = (indexToRemove) => {
    setWords((prevWords) => {
      const updated = prevWords.filter((_, idx) => idx !== indexToRemove);
      setMessageText(updated.join(" "));
      return updated;
    });
  };

  // Clear entire message
  const handleClearMessage = () => {
    setWords([]);
    setMessageText("");
    setTranslatedText("");
    setAudioBase64(null);
    setTranslationError(null);
    setSpeechError(null);
    if (audioInstanceRef.current) {
      audioInstanceRef.current.pause();
      audioInstanceRef.current = null;
    }
    setIsPlayingAudio(false);
  };

  // --- Camera & Recording Controls ---
  const startCamera = async () => {
    setCameraError(null);
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Webcam API not supported in this browser.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraActive(true);
    } catch (err) {
      setCameraError(err.message || "Failed to start camera. Check browser permissions.");
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
      } catch {
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
    if (!streamRef.current || recording || recognitionLoading) return;

    setCameraError(null);
    chunksRef.current = [];
    setCountdown(3);

    const mimeType =
      typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported("video/webm")
        ? "video/webm"
        : "video/mp4";

    try {
      const recorder = new MediaRecorder(streamRef.current, { mimeType });
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      recorder.onstop = async () => {
        setRecording(false);
        if (chunksRef.current.length === 0) {
          setCameraError("No video captured. Please try recording again.");
          return;
        }
        const blob = new Blob(chunksRef.current, { type: mimeType });
        await executeRecognition(blob, "webcam_gesture.webm");
      };

      mediaRecorderRef.current = recorder;
      recorder.start(100);
      setRecording(true);

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
      setCameraError(`Camera recording failed: ${err.message}`);
      setRecording(false);
    }
  };

  const executeRecognition = async (videoBlob, fileName) => {
    setRecognitionLoading(true);
    setCameraError(null);
    try {
      const response = await recognizeIsolatedVideo(videoBlob, fileName, null, 3);
      setRecognitionResult(response);

      // Add to message builder ONLY if recognition succeeds with confidence >= threshold
      const threshold = response.rejection_threshold || 0.80;
      if (response.status === "recognized" && response.is_valid && response.confidence >= threshold) {
        const recognizedSign = response.display_name || response.predicted_sign;
        handleAddWord(recognizedSign, Date.now());
      } else if (
        response.predicted_sign &&
        response.predicted_sign !== "unknown" &&
        response.predicted_sign !== "uncertain"
      ) {
        setUncertainSuggestion(response.display_name || response.predicted_sign);
      }
    } catch (err) {
      setCameraError(err.message || "Sign recognition request failed.");
      setRecognitionResult(null);
    } finally {
      setRecognitionLoading(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      executeRecognition(file, file.name);
    }
  };

  // --- Translation Action ---
  const handleTranslate = async () => {
    const textToTranslate = messageText.trim();
    if (!textToTranslate) {
      setTranslationError("Message is empty. Add or type words before translating.");
      return;
    }

    setTranslating(true);
    setTranslationError(null);
    setSpeechError(null);
    setAudioBase64(null);

    try {
      const data = await translateText(textToTranslate, targetLanguage, "en-IN");
      setTranslatedText(data.translated_text || textToTranslate);
    } catch (err) {
      setTranslationError(err.message || "Translation failed. Check Sarvam API key in backend.");
    } finally {
      setTranslating(false);
    }
  };

  // --- Text-to-Speech Action ---
  const handleGenerateSpeech = async () => {
    const textToSpeak = (translatedText || messageText).trim();
    if (!textToSpeak) {
      setSpeechError("No text available to convert to speech.");
      return;
    }

    setSynthesizing(true);
    setSpeechError(null);

    try {
      const data = await synthesizeSpeech(textToSpeak, targetLanguage, selectedVoice, 1.0);
      setAudioBase64(data.audio_base64);
      playAudio(data.audio_base64);
    } catch (err) {
      setSpeechError(err.message || "Failed to generate speech. Check Sarvam API key in backend.");
    } finally {
      setSynthesizing(false);
    }
  };

  // --- Audio Player Controls ---
  const playAudio = (base64Data) => {
    const audioData = base64Data || audioBase64;
    if (!audioData) return;

    if (audioInstanceRef.current) {
      audioInstanceRef.current.pause();
      audioInstanceRef.current = null;
    }

    const audioUrl = `data:audio/wav;base64,${audioData}`;
    const audio = new Audio(audioUrl);
    audioInstanceRef.current = audio;

    audio.onplay = () => setIsPlayingAudio(true);
    audio.onended = () => setIsPlayingAudio(false);
    audio.onerror = () => {
      setIsPlayingAudio(false);
      setSpeechError("Audio playback error.");
    };

    audio.play().catch((err) => {
      setIsPlayingAudio(false);
      setSpeechError(`Playback blocked: ${err.message}`);
    });
  };

  const handleStopAudio = () => {
    if (audioInstanceRef.current) {
      audioInstanceRef.current.pause();
      audioInstanceRef.current.currentTime = 0;
    }
    setIsPlayingAudio(false);
  };

  const handleReplay = () => {
    if (audioBase64) {
      playAudio(audioBase64);
    }
  };

  const renderUncertainBanner = () => {
    if (!uncertainSuggestion) return null;
    return (
      <div
        data-testid="uncertain-suggestion-banner"
        role="alert"
        style={{
          padding: "12px 16px",
          backgroundColor: "rgba(245, 158, 11, 0.15)",
          border: "1px solid #f59e0b",
          borderRadius: "8px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "10px",
          color: "#fbbf24",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <AlertCircle size={18} />
          <span>
            Uncertain detection: <strong>&ldquo;{uncertainSuggestion}&rdquo;</strong>?
          </span>
        </div>
        <div style={{ display: "flex", gap: "8px" }}>
          <button
            type="button"
            onClick={() => {
              handleAddWord(uncertainSuggestion, Date.now());
              setUncertainSuggestion(null);
            }}
            data-testid="add-uncertain-suggestion-btn"
            style={{
              padding: "5px 12px",
              backgroundColor: "#f59e0b",
              color: "#000",
              border: "none",
              borderRadius: "4px",
              fontWeight: "600",
              cursor: "pointer",
              fontSize: "0.85rem",
            }}
          >
            Add to Message
          </button>
          <button
            type="button"
            onClick={() => setUncertainSuggestion(null)}
            data-testid="dismiss-uncertain-suggestion-btn"
            style={{
              padding: "5px 12px",
              backgroundColor: "transparent",
              color: "#d1d5db",
              border: "1px solid #4b5563",
              borderRadius: "4px",
              cursor: "pointer",
              fontSize: "0.85rem",
            }}
          >
            Dismiss
          </button>
        </div>
      </div>
    );
  };

  const renderQuickAddPalette = () => (
    <div style={{ marginTop: "6px" }}>
      <span style={{ fontSize: "0.8rem", color: "#aaa", display: "block", marginBottom: "8px" }}>
        Quick-Add Signs (10 Supported ISL Words):
      </span>
      <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
        {SUPPORTED_SIGNS.map((sign) => (
          <button
            key={sign}
            type="button"
            onClick={() => handleAddWord(sign)}
            data-testid={`quick-add-${sign.toLowerCase().replace(/\s+/g, "-")}`}
            style={{
              padding: "5px 10px",
              backgroundColor: "#2a2a2a",
              border: "1px solid #444",
              borderRadius: "16px",
              color: "#fff",
              fontSize: "0.82rem",
              cursor: "pointer",
              display: "inline-flex",
              alignItems: "center",
              gap: "4px",
            }}
          >
            <PlusCircle size={12} style={{ color: "#64b5f6" }} />
            <span>{sign}</span>
          </button>
        ))}
      </div>
    </div>
  );

  const renderMessageAndSpeechPanels = () => (
    <section
      aria-label="Message Building, Translation and Speech Panel"
      style={{
        backgroundColor: "var(--color-surface, #1e1e1e)",
        border: "1px solid var(--color-border, #333)",
        borderRadius: "8px",
        padding: "20px",
        display: "flex",
        flexDirection: "column",
        gap: "20px",
      }}
    >
      {/* Section 2: Editable Message Builder */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
          <h2 style={{ margin: 0, fontSize: "1.2rem", color: "var(--color-primary, #90caf9)" }}>
            2. Message Builder
          </h2>
          {words.length > 0 && (
            <button
              type="button"
              onClick={handleClearMessage}
              data-testid="clear-message-btn"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
                padding: "4px 10px",
                backgroundColor: "transparent",
                border: "1px solid #555",
                borderRadius: "4px",
                color: "#f44336",
                fontSize: "0.8rem",
                cursor: "pointer",
              }}
            >
              <Trash2 size={13} />
              <span>Clear All</span>
            </button>
          )}
        </div>

        {/* Word Chips Row */}
        <div
          style={{
            minHeight: "44px",
            padding: "8px",
            backgroundColor: "#161616",
            border: "1px solid #333",
            borderRadius: "6px",
            display: "flex",
            flexWrap: "wrap",
            gap: "6px",
            alignItems: "center",
            marginBottom: "10px",
          }}
          data-testid="word-chips-container"
        >
          {words.length === 0 ? (
            <span style={{ fontSize: "0.85rem", color: "#666", fontStyle: "italic" }}>
              No recognized signs yet. Perform a sign or click quick-add buttons.
            </span>
          ) : (
            words.map((word, idx) => (
              <span
                key={`${word}-${idx}`}
                data-testid={`word-chip-${idx}`}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  padding: "4px 10px",
                  backgroundColor: "#1e3a5f",
                  border: "1px solid #2196f3",
                  borderRadius: "16px",
                  color: "#fff",
                  fontSize: "0.85rem",
                  fontWeight: "600",
                }}
              >
                <span>{word}</span>
                <button
                  type="button"
                  onClick={() => handleRemoveWord(idx)}
                  data-testid={`remove-word-${idx}`}
                  aria-label={`Remove word ${word}`}
                  style={{
                    background: "none",
                    border: "none",
                    color: "#90caf9",
                    cursor: "pointer",
                    padding: 0,
                    display: "flex",
                    alignItems: "center",
                  }}
                >
                  <X size={14} />
                </button>
              </span>
            ))
          )}
        </div>

        {/* Editable Textarea */}
        <div>
          <label
            htmlFor="editable-message-input"
            style={{ fontSize: "0.8rem", color: "#aaa", display: "block", marginBottom: "4px" }}
          >
            Editable Text (Deaf user can modify or add punctuation before translating):
          </label>
          <textarea
            id="editable-message-input"
            data-testid="editable-message-input"
            value={messageText}
            onChange={(e) => setMessageText(e.target.value)}
            placeholder="Editable sign message appears here..."
            rows={2}
            style={{
              width: "100%",
              padding: "10px",
              backgroundColor: "#262626",
              border: "1px solid #444",
              borderRadius: "6px",
              color: "#fff",
              fontSize: "1rem",
              resize: "vertical",
              boxSizing: "border-box",
            }}
          />
        </div>
      </div>

      {/* Section 3: Translation */}
      <div style={{ borderTop: "1px solid #333", paddingTop: "16px" }}>
        <h2 style={{ margin: "0 0 10px 0", fontSize: "1.2rem", color: "var(--color-primary, #90caf9)" }}>
          3. Language Selection & Translation
        </h2>

        <div style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap", marginBottom: "12px" }}>
          <label htmlFor="target-lang-select" style={{ fontSize: "0.85rem", color: "#aaa" }}>
            Target Language:
          </label>
          <select
            id="target-lang-select"
            data-testid="target-language-select"
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
            disabled={translating}
            style={{
              padding: "6px 12px",
              backgroundColor: "#262626",
              color: "#fff",
              border: "1px solid #444",
              borderRadius: "4px",
              fontSize: "0.85rem",
              flex: 1,
            }}
          >
            {Object.entries(availableLanguages).map(([code, name]) => (
              <option key={code} value={code}>
                {name} ({code})
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={handleTranslate}
            disabled={translating || !messageText.trim()}
            data-testid="translate-btn"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              padding: "8px 16px",
              backgroundColor: "#9c27b0",
              color: "#fff",
              border: "none",
              borderRadius: "6px",
              cursor: !messageText.trim() ? "not-allowed" : "pointer",
              fontWeight: "bold",
              opacity: !messageText.trim() ? 0.6 : 1,
            }}
          >
            {translating ? <Loader2 size={15} className="rotating" /> : <Languages size={15} />}
            <span>{translating ? "Translating..." : "Translate Message"}</span>
          </button>
        </div>

        {/* Translation Output Card */}
        <div
          style={{
            minHeight: "60px",
            padding: "12px 14px",
            backgroundColor: "#161616",
            border: "1px solid #333",
            borderRadius: "6px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
          }}
          data-testid="translation-result-box"
        >
          {translating ? (
            <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#ce93d8" }}>
              <Loader2 size={16} className="rotating" />
              <span style={{ fontSize: "0.9rem" }}>Translating message with Sarvam AI...</span>
            </div>
          ) : translationError ? (
            <div style={{ color: "#ef5350", fontSize: "0.85rem" }}>
              <strong>Translation Error:</strong> {translationError}
            </div>
          ) : translatedText ? (
            <div>
              <span style={{ fontSize: "0.75rem", color: "#888", textTransform: "uppercase" }}>
                Translated Message ({availableLanguages[targetLanguage] || targetLanguage}):
              </span>
              <p
                data-testid="translated-text-display"
                style={{
                  margin: "4px 0 0 0",
                  fontSize: "1.25rem",
                  fontWeight: "700",
                  color: "#fff",
                }}
              >
                "{translatedText}"
              </p>
            </div>
          ) : (
            <span style={{ fontSize: "0.85rem", color: "#666" }}>
              Click "Translate Message" to translate into the hearing person's language.
            </span>
          )}
        </div>
      </div>

      {/* Section 4: Text-to-Speech (Bulbul) */}
      <div style={{ borderTop: "1px solid #333", paddingTop: "16px" }}>
        <h2 style={{ margin: "0 0 10px 0", fontSize: "1.2rem", color: "var(--color-primary, #90caf9)" }}>
          4. Spoken Audio (Sarvam Bulbul TTS)
        </h2>

        <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap", marginBottom: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <label htmlFor="voice-select" style={{ fontSize: "0.85rem", color: "#aaa" }}>
              Voice:
            </label>
            <select
              id="voice-select"
              data-testid="voice-select"
              value={selectedVoice}
              onChange={(e) => setSelectedVoice(e.target.value)}
              disabled={synthesizing}
              style={{
                padding: "6px 8px",
                backgroundColor: "#262626",
                color: "#fff",
                border: "1px solid #444",
                borderRadius: "4px",
                fontSize: "0.85rem",
              }}
            >
              {SUPPORTED_VOICES.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.label}
                </option>
              ))}
            </select>
          </div>

          <button
            type="button"
            onClick={handleGenerateSpeech}
            disabled={synthesizing || (!translatedText && !messageText.trim())}
            data-testid="generate-speech-btn"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              padding: "8px 18px",
              backgroundColor: "#ff9800",
              color: "#000",
              border: "none",
              borderRadius: "6px",
              cursor: !translatedText && !messageText.trim() ? "not-allowed" : "pointer",
              fontWeight: "bold",
              opacity: !translatedText && !messageText.trim() ? 0.6 : 1,
            }}
          >
            {synthesizing ? <Loader2 size={16} className="rotating" /> : <Volume2 size={16} />}
            <span>{synthesizing ? "Synthesizing..." : "Generate Speech"}</span>
          </button>
        </div>

        {/* Playback Controls & Status */}
        <div
          style={{
            padding: "12px 14px",
            backgroundColor: "#161616",
            border: "1px solid #333",
            borderRadius: "6px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
          data-testid="speech-playback-panel"
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            {audioBase64 ? (
              <>
                <button
                  type="button"
                  onClick={handleReplay}
                  disabled={isPlayingAudio}
                  data-testid="replay-audio-btn"
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "6px",
                    padding: "6px 14px",
                    backgroundColor: "#4caf50",
                    color: "#fff",
                    border: "none",
                    borderRadius: "4px",
                    cursor: "pointer",
                    fontWeight: "600",
                    fontSize: "0.85rem",
                  }}
                >
                  <RotateCcw size={14} />
                  <span>{isPlayingAudio ? "Playing..." : "Play Audio"}</span>
                </button>

                {isPlayingAudio && (
                  <button
                    type="button"
                    onClick={handleStopAudio}
                    data-testid="stop-audio-btn"
                    style={{
                      padding: "6px 12px",
                      backgroundColor: "#f44336",
                      color: "#fff",
                      border: "none",
                      borderRadius: "4px",
                      cursor: "pointer",
                      fontSize: "0.85rem",
                    }}
                  >
                    Stop
                  </button>
                )}

                <span style={{ fontSize: "0.8rem", color: isPlayingAudio ? "#81c784" : "#aaa" }}>
                  {isPlayingAudio ? "🔊 Speaking aloud" : "✓ Audio ready"}
                </span>
              </>
            ) : (
              <span style={{ fontSize: "0.85rem", color: "#666" }}>
                No audio generated yet. Click "Generate Speech".
              </span>
            )}
          </div>

          {audioBase64 && (
            <span style={{ fontSize: "0.78rem", color: "#777" }}>
              Voice: {selectedVoice.toUpperCase()} &bull; Bulbul TTS
            </span>
          )}
        </div>

        {speechError && (
          <div
            style={{
              marginTop: "8px",
              color: "#ef5350",
              fontSize: "0.82rem",
              backgroundColor: "rgba(239, 83, 80, 0.1)",
              padding: "8px 12px",
              borderRadius: "4px",
            }}
          >
            ⚠️ {speechError}
          </div>
        )}
      </div>
    </section>
  );

  return (
    <div
      className="isl-message-builder-container"
      data-testid="isl-message-builder"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "24px",
      }}
    >
      {/* Workflow Navigation Banner */}
      <div
        style={{
          padding: "12px 18px",
          backgroundColor: "var(--color-surface, #1e1e1e)",
          border: "1px solid var(--color-border, #333)",
          borderRadius: "8px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "10px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ fontSize: "1.4rem" }}>🤟</span>
          <div>
            <strong style={{ color: "var(--color-primary, #64b5f6)" }}>
              FE-08 Complete Direction:
            </strong>{" "}
            <span style={{ color: "var(--color-text, #eee)" }}>
              Deaf Person (ISL Signs) &rarr; Editable Message &rarr; Sarvam Translation &rarr; Bulbul Speech
            </span>
          </div>
        </div>
        <div style={{ fontSize: "0.82rem", color: "var(--color-text-muted, #888)" }}>
          10-Class Bi-GRU Model &bull; 0.80 Confidence Gate
        </div>
      </div>

      {hideCameraPreview ? (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {renderUncertainBanner()}
          <div
            style={{
              backgroundColor: "var(--color-surface, #1e1e1e)",
              border: "1px solid var(--color-border, #333)",
              borderRadius: "8px",
              padding: "16px 20px",
            }}
          >
            {renderQuickAddPalette()}
          </div>
          {renderMessageAndSpeechPanels()}
        </div>
      ) : (
        <>
          {renderUncertainBanner()}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
            {/* ======================================================== */}
            {/* LEFT COLUMN: ISL Gesture Recognition & Camera Workspace */}
            {/* ======================================================== */}
            <section
              aria-label="ISL Recognition Workspace"
              style={{
                backgroundColor: "var(--color-surface, #1e1e1e)",
                border: "1px solid var(--color-border, #333)",
                borderRadius: "8px",
                padding: "20px",
                display: "flex",
                flexDirection: "column",
                gap: "16px",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <h2 style={{ margin: 0, fontSize: "1.2rem", color: "var(--color-primary, #90caf9)" }}>
                  1. ISL Gesture Recognition
                </h2>
                <span
                  style={{
                    fontSize: "0.75rem",
                    padding: "3px 8px",
                    borderRadius: "12px",
                    backgroundColor: cameraActive ? "#1b5e20" : "#333",
                    color: cameraActive ? "#a5d6a7" : "#aaa",
                    fontWeight: "600",
                  }}
                >
                  {cameraActive ? "Camera Live" : "Camera Idle"}
                </span>
              </div>

              {/* Video Viewport */}
              <div
                style={{
                  position: "relative",
                  width: "100%",
                  height: "280px",
                  backgroundColor: "#111",
                  borderRadius: "6px",
                  overflow: "hidden",
                  border: recording ? "3px solid #f44336" : "1px solid #333",
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
                    transform: "scaleX(-1)",
                    display: cameraActive ? "block" : "none",
                  }}
                />

                {!cameraActive && (
                  <div style={{ textAlign: "center", color: "#888", padding: "20px" }}>
                    <p style={{ margin: "0 0 6px 0", fontWeight: "600" }}>Camera Off</p>
                    <p style={{ margin: 0, fontSize: "0.82rem" }}>
                      Start camera to capture signs, or use quick-add buttons below.
                    </p>
                  </div>
                )}

                {/* Recording Countdown Badge */}
                {recording && (
                  <div
                    style={{
                      position: "absolute",
                      top: "12px",
                      left: "12px",
                      backgroundColor: "rgba(244, 67, 54, 0.9)",
                      color: "#fff",
                      padding: "4px 10px",
                      borderRadius: "16px",
                      fontSize: "0.85rem",
                      fontWeight: "bold",
                    }}
                  >
                    Recording: {countdown}s
                  </div>
                )}

                {/* Inference Spinner Overlay */}
                {recognitionLoading && (
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      backgroundColor: "rgba(0,0,0,0.75)",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "#90caf9",
                      gap: "8px",
                    }}
                  >
                    <Loader2 size={32} className="rotating" />
                    <span style={{ fontSize: "0.9rem", fontWeight: "600" }}>
                      Classifying Gesture with Bi-GRU...
                    </span>
                  </div>
                )}
              </div>

              {/* Camera Buttons */}
              <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "center" }}>
                {!cameraActive ? (
                  <button
                    type="button"
                    onClick={startCamera}
                    disabled={recognitionLoading}
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "6px",
                      padding: "8px 16px",
                      backgroundColor: "#2196f3",
                      color: "#fff",
                      border: "none",
                      borderRadius: "6px",
                      cursor: "pointer",
                      fontWeight: "600",
                    }}
                  >
                    <Video size={16} />
                    <span>Start Camera</span>
                  </button>
                ) : (
                  <>
                    <button
                      type="button"
                      onClick={stopCamera}
                      disabled={recording || recognitionLoading}
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "6px",
                        padding: "8px 14px",
                        backgroundColor: "#424242",
                        color: "#eee",
                        border: "none",
                        borderRadius: "6px",
                        cursor: "pointer",
                      }}
                    >
                      <VideoOff size={16} />
                      <span>Stop</span>
                    </button>

                    <button
                      type="button"
                      onClick={captureAndRecognize}
                      disabled={recording || recognitionLoading}
                      style={{
                        padding: "8px 20px",
                        backgroundColor: recording ? "#f44336" : "#4caf50",
                        color: "#fff",
                        border: "none",
                        borderRadius: "6px",
                        cursor: "pointer",
                        fontWeight: "bold",
                      }}
                    >
                      {recording ? `Recording (${countdown}s)...` : "Capture Gesture (3s)"}
                    </button>
                  </>
                )}

                <div style={{ marginLeft: "auto" }}>
                  <label style={{ fontSize: "0.82rem", color: "#888", cursor: "pointer", textDecoration: "underline" }}>
                    Upload test video
                    <input
                      type="file"
                      accept=".mp4,.webm,.avi"
                      onChange={handleFileUpload}
                      disabled={recognitionLoading || recording}
                      style={{ display: "none" }}
                    />
                  </label>
                </div>
              </div>

              {cameraError && (
                <div
                  style={{
                    backgroundColor: "rgba(244, 67, 54, 0.15)",
                    border: "1px solid #f44336",
                    borderRadius: "6px",
                    padding: "8px 12px",
                    fontSize: "0.82rem",
                    color: "#ffcdd2",
                  }}
                >
                  ⚠️ {cameraError}
                </div>
              )}

              {/* Last Recognition Feedback Box */}
              {recognitionResult && (
                <div
                  style={{
                    backgroundColor: "#161616",
                    border: "1px solid #333",
                    borderRadius: "6px",
                    padding: "10px 14px",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: "0.75rem", color: "#888" }}>LAST RECOGNITION</span>
                    <span
                      style={{
                        fontSize: "0.75rem",
                        padding: "2px 8px",
                        borderRadius: "10px",
                        fontWeight: "bold",
                        backgroundColor:
                          recognitionResult.status === "recognized" ? "#1b5e20" : "#e65100",
                        color: recognitionResult.status === "recognized" ? "#a5d6a7" : "#ffe0b2",
                      }}
                    >
                      {recognitionResult.status === "recognized" ? "✓ ADDED TO MESSAGE" : "⚠ REJECTED (UNCERTAIN)"}
                    </span>
                  </div>
                  <div style={{ display: "flex", alignItems: "baseline", gap: "10px", marginTop: "4px" }}>
                    <strong style={{ fontSize: "1.4rem", textTransform: "uppercase" }}>
                      {recognitionResult.display_name || recognitionResult.predicted_sign}
                    </strong>
                    <span style={{ fontSize: "0.95rem", color: "#81c784" }}>
                      {(recognitionResult.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  {recognitionResult.status !== "recognized" && (
                    <div style={{ fontSize: "0.78rem", color: "#ffb74d", marginTop: "4px" }}>
                      Confidence below 0.80 threshold. Rejected to prevent corrupting the message.
                    </div>
                  )}
                </div>
              )}

              {renderQuickAddPalette()}
            </section>

            {/* ======================================================== */}
            {/* RIGHT COLUMN: Message Builder, Translation, & Speech UI */}
            {/* ======================================================== */}
            {renderMessageAndSpeechPanels()}
          </div>
        </>
      )}
    </div>
  );
}
