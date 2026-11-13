"""
Text-to-Speech (TTS) synthesizer for Signova AI.
Supports:
1. Windows SAPI via PowerShell System.Speech.Synthesis (synthesize to WAV or direct speak).
2. Fallback pure-Python PCM WAV generator for headless or non-Windows test environments.
"""

import logging
from pathlib import Path
import subprocess
import sys
import wave
from typing import Optional, Union

logger = logging.getLogger("signova.speech.tts")


class TextToSpeech:
    """Production Text-to-Speech synthesizer supporting audio export and live playback."""

    def __init__(self, voice_name: Optional[str] = None, rate: int = 0) -> None:
        self.voice_name = voice_name
        self.rate = rate  # -10 to +10 for SAPI

    def synthesize_to_wav(self, text: str, output_path: Union[str, Path]) -> Path:
        """
        Synthesize text into a .wav file.
        Uses Windows System.Speech if on Windows, falling back to clean PCM tone generation.
        """
        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        if sys.platform.startswith("win"):
            try:
                # Use Windows SAPI via PowerShell to render to WAV file
                escaped_text = text.replace("'", "''")
                escaped_path = str(output_file).replace("'", "''")
                ps_script = (
                    f"Add-Type -AssemblyName System.Speech; "
                    f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Rate = {self.rate}; "
                    f"$s.SetOutputToWaveFile('{escaped_path}'); "
                    f"$s.Speak('{escaped_text}'); "
                    f"$s.Dispose()"
                )
                res = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if res.returncode == 0 and output_file.exists() and output_file.stat().st_size > 44:
                    logger.info("Synthesized %d bytes of speech to %s", output_file.stat().st_size, output_file)
                    return output_file
                else:
                    logger.warning("Windows SAPI PowerShell synthesis returned: %s. Falling back to synthetic PCM wav.", res.stderr)
            except Exception as exc:
                logger.warning("Windows SAPI synthesis error: %s. Falling back to synthetic PCM wav.", exc)

        # Standalone WAV header generator fallback for test/headless environments
        return self._create_synthetic_wav(output_file, duration_sec=1.0)

    def speak(self, text: str) -> bool:
        """
        Speak text immediately via system speaker.
        Returns True if successful, False if no audio device / headless.
        """
        if not text or not text.strip():
            return False

        if sys.platform.startswith("win"):
            try:
                escaped_text = text.replace("'", "''")
                ps_script = (
                    f"Add-Type -AssemblyName System.Speech; "
                    f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Rate = {self.rate}; "
                    f"$s.Speak('{escaped_text}'); "
                    f"$s.Dispose()"
                )
                res = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                return res.returncode == 0
            except Exception as exc:
                logger.warning("Direct speech failed: %s", exc)
                return False

        return False

    @staticmethod
    def _create_synthetic_wav(output_path: Path, duration_sec: float = 1.0, sample_rate: int = 16000) -> Path:
        """Create a valid silence/carrier WAV file for test pipelines."""
        import struct

        num_samples = int(duration_sec * sample_rate)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            raw_data = struct.pack(f"<{num_samples}h", *([0] * num_samples))
            wf.writeframes(raw_data)

        return output_path


# Global instance
text_to_speech = TextToSpeech()
