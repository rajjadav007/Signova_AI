import React, { useEffect, useState } from "react";

/**
 * Text-to-Speech component using native Web Speech API.
 * Provides speak, stop, and auto-speak toggle controls.
 */
export default function SpeechSynthesizer({ text, autoSpeak = false }) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(null);

  useEffect(() => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      setIsSupported(true);

      const updateVoices = () => {
        const available = window.speechSynthesis.getVoices();
        setVoices(available);
        if (available.length > 0 && !selectedVoice) {
          // Default to first English voice or first available
          const enVoice = available.find((v) => v.lang.startsWith("en")) || available[0];
          setSelectedVoice(enVoice.name);
        }
      };

      updateVoices();
      window.speechSynthesis.onvoiceschanged = updateVoices;
    }
  }, []);

  const speak = (content) => {
    if (!isSupported || !content) return;

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(content);
    if (selectedVoice) {
      const voiceObj = voices.find((v) => v.name === selectedVoice);
      if (voiceObj) utterance.voice = voiceObj;
    }

    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  };

  const stop = () => {
    if (isSupported) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  };

  useEffect(() => {
    if (autoSpeak && text && text.trim().length > 0) {
      speak(text);
    }
  }, [text, autoSpeak]);

  if (!isSupported) {
    return (
      <div className="tts-container fallback">
        <small>Text-to-Speech is not supported in this browser.</small>
      </div>
    );
  }

  return (
    <div className="tts-controls" style={{ display: "flex", gap: "8px", alignItems: "center" }}>
      <button
        type="button"
        onClick={() => speak(text)}
        disabled={!text || isSpeaking}
        style={{
          padding: "6px 14px",
          backgroundColor: isSpeaking ? "#4caf50" : "#2196f3",
          color: "white",
          border: "none",
          borderRadius: "4px",
          cursor: !text ? "not-allowed" : "pointer",
        }}
      >
        {isSpeaking ? "Speaking..." : "Speak Text"}
      </button>

      {isSpeaking && (
        <button
          type="button"
          onClick={stop}
          style={{
            padding: "6px 12px",
            backgroundColor: "#f44336",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          Stop
        </button>
      )}

      {voices.length > 1 && (
        <select
          value={selectedVoice || ""}
          onChange={(e) => setSelectedVoice(e.target.value)}
          style={{ padding: "6px 8px", borderRadius: "4px" }}
          aria-label="Select voice"
        >
          {voices.map((v) => (
            <option key={v.name} value={v.name}>
              {v.name} ({v.lang})
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
