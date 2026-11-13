import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import React from "react";
import { BrowserRouter } from "react-router-dom";
import DashboardPage from "./DashboardPage";
import * as api from "../services/api";

function renderWithRouter(ui) {
  return render(<BrowserRouter>{ui}</BrowserRouter>);
}

describe("Signova AI Dashboard Page (FE-03)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(api, "fetchHealth").mockResolvedValue({
      status: "healthy",
      application: "Signova AI Backend",
      version: "0.1.0",
      environment: "test",
    });
  });

  it("renders Welcome section with exact title, description, and action buttons", async () => {
    renderWithRouter(<DashboardPage />);

    expect(screen.getByRole("heading", { name: "Welcome to Signova AI" })).toBeDefined();
    expect(
      screen.getByText("Breaking communication barriers through Indian Sign Language recognition.")
    ).toBeDefined();
    expect(
      screen.getByText(/Signova AI translates Indian Sign Language \(ISL\) gestures into real-time text and speech/i)
    ).toBeDefined();

    const startBtn = screen.getByRole("link", { name: /Start Live Recognition/i });
    expect(startBtn.getAttribute("href")).toBe("/recognition");

    const exploreBtn = screen.getByRole("link", { name: /Explore Vocabulary/i });
    expect(exploreBtn.getAttribute("href")).toBe("/vocabulary");

    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeDefined();
    });
  });

  it("displays system overview cards with accurate information", async () => {
    renderWithRouter(<DashboardPage />);

    // Supported Signs
    const signsCard = screen.getByTestId("card-supported-signs");
    expect(within(signsCard).getByText("10")).toBeDefined();

    // Recognition Model
    const modelCard = screen.getByTestId("card-recognition-model");
    expect(within(modelCard).getByText("Bi-GRU")).toBeDefined();

    // Recognition Type
    const typeCard = screen.getByTestId("card-recognition-type");
    expect(within(typeCard).getByText("Isolated Sign")).toBeDefined();

    // Backend Service
    const backendCard = screen.getByTestId("card-backend-service");
    await waitFor(() => {
      expect(within(backendCard).getByText("Connected")).toBeDefined();
    });
    expect(within(backendCard).getByText(/Signova AI Backend \(v0.1.0\)/i)).toBeDefined();
  });

  it("handles disconnected backend status and allows refresh retry", async () => {
    vi.spyOn(api, "fetchHealth").mockRejectedValueOnce(new Error("Connection refused"));

    renderWithRouter(<DashboardPage />);

    const backendCard = screen.getByTestId("card-backend-service");
    await waitFor(() => {
      expect(within(backendCard).getByText("Disconnected")).toBeDefined();
    });
    expect(within(backendCard).getByText(/API unreachable \(port 8000\)/i)).toBeDefined();

    // Mock successful reconnect
    vi.spyOn(api, "fetchHealth").mockResolvedValueOnce({
      status: "healthy",
      application: "Signova AI Backend",
      version: "0.1.0",
    });

    const refreshBtn = screen.getByRole("button", { name: /Refresh backend status/i });
    fireEvent.click(refreshBtn);

    await waitFor(() => {
      expect(within(backendCard).getByText("Connected")).toBeDefined();
    });
  });

  it("displays model architecture configuration details and notice", async () => {
    renderWithRouter(<DashboardPage />);

    expect(screen.getByRole("heading", { name: "Model Architecture" })).toBeDefined();
    expect(screen.getByText("Configuration Spec")).toBeDefined();

    expect(screen.getByText("Architecture")).toBeDefined();
    expect(screen.getByText("2-layer Bidirectional GRU")).toBeDefined();
    expect(screen.getByText("Hidden Dimension")).toBeDefined();
    expect(screen.getByText("128")).toBeDefined();
    expect(screen.getByText("Input Features")).toBeDefined();
    expect(screen.getByText("600")).toBeDefined();
    expect(screen.getByText("Temporal Sequence Length")).toBeDefined();
    expect(screen.getByText("64")).toBeDefined();
    expect(screen.getByText("Supported Classes")).toBeDefined();

    expect(
      screen.getByText(/Weights are loaded on demand by the backend inference service/i)
    ).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeDefined();
    });
  });

  it("displays the 10 supported signs and View All Signs navigation link", async () => {
    renderWithRouter(<DashboardPage />);

    expect(screen.getByRole("heading", { name: "Supported Vocabulary" })).toBeDefined();

    const expectedSigns = [
      "Hello",
      "Thank you",
      "Sorry",
      "Please",
      "Yes",
      "No",
      "Help",
      "Stop",
      "Water",
      "Food",
    ];

    expectedSigns.forEach((sign) => {
      expect(screen.getByText(sign)).toBeDefined();
    });

    const viewAllLinks = screen.getAllByRole("link", { name: /View All Signs/i });
    expect(viewAllLinks.length).toBeGreaterThanOrEqual(1);
    expect(viewAllLinks[0].getAttribute("href")).toBe("/vocabulary");

    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeDefined();
    });
  });

  it("displays visual How It Works pipeline and MVP scope banner", async () => {
    renderWithRouter(<DashboardPage />);

    expect(screen.getByRole("heading", { name: "How It Works" })).toBeDefined();
    expect(screen.getByText("Webcam / Video")).toBeDefined();
    expect(screen.getByText("MediaPipe Landmarks")).toBeDefined();
    expect(screen.getByText("Feature Engineering")).toBeDefined();
    expect(screen.getByText("Bi-GRU Recognition")).toBeDefined();
    expect(screen.getByText("Prediction")).toBeDefined();
    expect(screen.getByText("Text & Speech")).toBeDefined();

    expect(
      screen.getByText(/The current system operates as an isolated sign recognizer using fixed 3-second capture windows/i)
    ).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeDefined();
    });
  });

  it("displays Recognition Engine Overview features", async () => {
    renderWithRouter(<DashboardPage />);

    expect(screen.getByRole("heading", { name: "Recognition Engine Overview" })).toBeDefined();
    expect(screen.getByText("Primary Recognition Engine")).toBeDefined();
    expect(screen.getByText("10 Supported Classes")).toBeDefined();
    expect(screen.getByText("Capture-and-Recognize")).toBeDefined();
    expect(screen.getByText("Confidence Rejection")).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeDefined();
    });
  });
});
