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
} from "lucide-react";
import LiveWebcamPreview from "../camera/LiveWebcamPreview";
import { translateText, synthesizeSpeech } from "../../services/api";

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
}) {
  const [currentSign, setCurrentSign] = useState(null);
  const [translatedText, setTranslatedText] = useState("");
  const [isTranslating, setIsTranslating] = useState(false);
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [audioBlocked, setAudioBlocked] = useState(false);
  const [uncertainSign, setUncertainSign] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const audioRef = useRef(null);
  const lastProcessedSignRef = useRef({ sign: null, timestamp: 0 });
  const activeLangRef = useRef(otherLanguage);

  useEffect(() => {
    activeLangRef.current = otherLanguage;
    // Re-translate current sign if language selection changed
    if (currentSign) {
      handleTranslateAndSpeak(currentSign, otherLanguage);
    }
  }, [otherLanguage]);

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);

  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setIsPlayingAudio(false);
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
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
      if (!signName) return;
      const cleanSign =
        VOCABULARY_SIGNS.find(
          (v) => v.toLowerCase() === signName.trim().toLowerCase()
        ) || signName.trim();

      setCurrentSign(cleanSign);
      setErrorMessage(null);

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
          translated = res.translated_text || cleanSign;
        } catch (err) {
          translated = cleanSign;
          setErrorMessage(err.message || "Translation service unavailable; using original label.");
        } finally {
          setIsTranslating(false);
        }
      }

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
        if (ttsRes && ttsRes.audio_base64) {
          audioCache.set(cacheKey, ttsRes.audio_base64);
          playBase64Audio(ttsRes.audio_base64);
        } else {
          speakWithBrowserFallback(translated, targetLang);
        }
      } catch {
        // Fallback gracefully without throwing
        speakWithBrowserFallback(translated, targetLang);
      } finally {
        setIsSynthesizing(false);
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
    setTranslatedText("");
    setUncertainSign(null);
    setErrorMessage(null);
    setAudioBlocked(false);
  };

  const replayAudio = () => {
    if (!translatedText) return;
    const cacheKey = `${translatedText}_${otherLanguage}`;
    if (audioCache.has(cacheKey)) {
      playBase64Audio(audioCache.get(cacheKey));
    } else {
      handleTranslateAndSpeak(currentSign || translatedText, otherLanguage);
    }
  };

  return (
    <div
      className="communication-panel panel-your-sign"
      data-testid="panel-your-sign"
      aria-label="Direction A: Your Sign to Spoken Audio"
    >
      <header className="panel-header">
        <div className="panel-title-group">
          <span className="panel-direction-badge deaf-badge">Deaf Person → Hearing Person</span>
          <h2 className="panel-title">Your Sign</h2>
        </div>
        <span className="panel-target-lang-pill" data-testid="panel-a-target-lang">
          Speaking: <strong>{otherLanguageName}</strong>
        </span>
      </header>

      {/* Webcam Feed (Single Camera Instance, initialAutoSpeak false to route through Sarvam TTS) */}
      <div className="panel-camera-section">
        <LiveWebcamPreview
          onSignRecognized={handleSignRecognized}
          onUncertainSuggestion={handleUncertainSuggestion}
          initialAutoSpeak={false}
          showSpeechToggle={false}
        />
      </div>

      {/* Uncertain Prediction Card (interactive confirm, correct, dismiss) */}
      {uncertainSign && (
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
      {currentSign && (
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
            <div className="panel-error-notice" role="alert">
              {errorMessage}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
