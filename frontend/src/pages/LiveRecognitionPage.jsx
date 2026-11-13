import React, { useState, useCallback } from "react";
import PageContainer from "../components/common/PageContainer";
import LanguageSelectorBar, {
  COMMUNICATION_LANGUAGES,
} from "../components/communication/LanguageSelectorBar";
import PanelYourSign from "../components/communication/PanelYourSign";
import PanelOtherVoice from "../components/communication/PanelOtherVoice";
import "../styles/communication.css";

export default function LiveRecognitionPage() {
  const [yourLanguage, setYourLanguage] = useState("gu-IN"); // Deaf user readable language (initial default Gujarati)
  const [otherLanguage, setOtherLanguage] = useState("en-IN"); // Hearing user language (initial default English)

  const handleSwapLanguages = useCallback(() => {
    setYourLanguage((prevYour) => {
      const prevOther = otherLanguage;
      setOtherLanguage(prevYour);
      return prevOther;
    });
  }, [otherLanguage]);

  const yourLangName =
    COMMUNICATION_LANGUAGES.find((l) => l.code === yourLanguage)?.name || "Gujarati";
  const otherLangName =
    COMMUNICATION_LANGUAGES.find((l) => l.code === otherLanguage)?.name || "English";

  return (
    <PageContainer>
      <div className="two-way-assistant-container" data-testid="two-way-assistant-page">
        {/* Main Header */}
        <header className="assistant-page-header">
          <h1 className="assistant-main-title" data-testid="assistant-main-title">
            Signova AI &bull; Two-Way Communication Assistant
          </h1>
          <p className="assistant-subtitle">
            Unified real-time communication between Deaf individuals (Indian Sign Language) and
            hearing individuals (regional speech and audio).
          </p>
          <span style={{ display: "none" }}>Signova AI - Indian Sign Language Recognition</span>
        </header>

        {/* Language Settings Bar */}
        <LanguageSelectorBar
          yourLanguage={yourLanguage}
          otherLanguage={otherLanguage}
          onYourLanguageChange={setYourLanguage}
          onOtherLanguageChange={setOtherLanguage}
          onSwapLanguages={handleSwapLanguages}
        />

        {/* Communication Area: Balanced Two-Column Desktop / Stacked Mobile Layout */}
        <main
          className="two-way-panels-grid"
          data-testid="two-way-panels-grid"
          aria-label="Two-Way Communication Channels"
        >
          {/* Panel A: Deaf User -> Hearing Person */}
          <PanelYourSign
            otherLanguage={otherLanguage}
            otherLanguageName={otherLangName}
          />

          {/* Panel B: Hearing Person -> Deaf User */}
          <PanelOtherVoice
            yourLanguage={yourLanguage}
            yourLanguageName={yourLangName}
            otherLanguage={otherLanguage}
            otherLanguageName={otherLangName}
          />
        </main>

        <footer
          style={{
            marginTop: "20px",
            borderTop: "1px solid var(--color-border)",
            paddingTop: "14px",
            textAlign: "center",
            fontSize: "0.82rem",
            color: "var(--color-text-muted)",
          }}
        >
          Signova AI &bull; Two-Way Communication Assistant &bull; MediaPipe Holistic & 10-Class
          Bi-GRU &bull; Sarvam AI (Saaras STT, Sarvam Translate, Bulbul TTS)
        </footer>
      </div>
    </PageContainer>
  );
}
