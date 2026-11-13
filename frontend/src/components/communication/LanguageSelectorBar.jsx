import React from "react";
import { ArrowLeftRight, Globe } from "lucide-react";

export const COMMUNICATION_LANGUAGES = [
  { code: "gu-IN", name: "Gujarati", nativeName: "ગુજરાતી" },
  { code: "hi-IN", name: "Hindi", nativeName: "हिन्दी" },
  { code: "en-IN", name: "English", nativeName: "English" },
  { code: "kn-IN", name: "Kannada", nativeName: "ಕನ್ನಡ" },
  { code: "ta-IN", name: "Tamil", nativeName: "தமிழ்" },
];

export default function LanguageSelectorBar({
  yourLanguage = "gu-IN",
  otherLanguage = "en-IN",
  onYourLanguageChange,
  onOtherLanguageChange,
  onSwapLanguages,
}) {
  const yourLangObj =
    COMMUNICATION_LANGUAGES.find((l) => l.code === yourLanguage) || COMMUNICATION_LANGUAGES[0];
  const otherLangObj =
    COMMUNICATION_LANGUAGES.find((l) => l.code === otherLanguage) || COMMUNICATION_LANGUAGES[2];

  return (
    <section
      className="language-selector-bar"
      aria-label="Two-Way Communication Language Settings"
      data-testid="language-selector-bar"
    >
      <div className="language-selector-container">
        {/* Selector A: Your Language (Deaf User) */}
        <div className="language-selector-group" data-testid="group-your-language">
          <div className="language-label-wrap">
            <span className="language-role-badge your-badge">Deaf User</span>
            <label htmlFor="your-language-select" className="language-selector-label">
              Your Language
            </label>
          </div>
          <div className="language-select-wrapper">
            <Globe size={18} className="language-globe-icon" aria-hidden="true" />
            <select
              id="your-language-select"
              data-testid="your-language-select"
              className="language-dropdown-select"
              value={yourLanguage}
              onChange={(e) => onYourLanguageChange && onYourLanguageChange(e.target.value)}
              aria-label="Your Language (Readable Regional Language)"
            >
              {COMMUNICATION_LANGUAGES.map((lang) => (
                <option key={`your-${lang.code}`} value={lang.code}>
                  {lang.name} ({lang.nativeName})
                </option>
              ))}
            </select>
          </div>
          <span className="language-active-hint" data-testid="your-language-active">
            Output text: <strong>{yourLangObj.name}</strong> ({yourLangObj.nativeName})
          </span>
        </div>

        {/* Swap Languages Control */}
        <div className="language-swap-wrap">
          <button
            type="button"
            className="language-swap-button"
            data-testid="swap-languages-btn"
            onClick={onSwapLanguages}
            aria-label="Swap communication languages"
            title="Swap Your Language and Other Person's Language"
          >
            <ArrowLeftRight size={18} />
            <span className="swap-btn-text">Swap</span>
          </button>
        </div>

        {/* Selector B: Other Person's Language (Hearing User) */}
        <div className="language-selector-group" data-testid="group-other-language">
          <div className="language-label-wrap">
            <span className="language-role-badge other-badge">Hearing Person</span>
            <label htmlFor="other-language-select" className="language-selector-label">
              Other Person's Language
            </label>
          </div>
          <div className="language-select-wrapper">
            <Globe size={18} className="language-globe-icon" aria-hidden="true" />
            <select
              id="other-language-select"
              data-testid="other-language-select"
              className="language-dropdown-select"
              value={otherLanguage}
              onChange={(e) => onOtherLanguageChange && onOtherLanguageChange(e.target.value)}
              aria-label="Other Person's Language (Spoken & Hearing Language)"
            >
              {COMMUNICATION_LANGUAGES.map((lang) => (
                <option key={`other-${lang.code}`} value={lang.code}>
                  {lang.name} ({lang.nativeName})
                </option>
              ))}
            </select>
          </div>
          <span className="language-active-hint" data-testid="other-language-active">
            Spoken speech: <strong>{otherLangObj.name}</strong> ({otherLangObj.nativeName})
          </span>
        </div>
      </div>
    </section>
  );
}
