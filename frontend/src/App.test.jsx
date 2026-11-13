import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import React from "react";
import App from "./App";
import * as api from "./services/api";

// Mock mediaDevices for webcam components
if (!navigator.mediaDevices) {
  navigator.mediaDevices = {};
}
navigator.mediaDevices.getUserMedia = vi.fn().mockResolvedValue({
  getTracks: () => [{ stop: vi.fn() }],
});

// Mock SpeechSynthesis
window.speechSynthesis = {
  speak: vi.fn(),
  cancel: vi.fn(),
  getVoices: vi.fn().mockReturnValue([
    { name: "Alex", lang: "en-US" },
    { name: "Samantha", lang: "en-US" },
  ]),
  onvoiceschanged: null,
};

describe("Signova AI Application Layout & Navigation (FE-02)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(api, "fetchHealth").mockResolvedValue({
      status: "healthy",
      application: "Signova AI Backend",
      version: "0.1.0",
      environment: "test",
    });
  });

  it("renders branding, sidebar navigation links, and default dashboard view", async () => {
    render(<App />);

    // Branding (found in sidebar brand and navbar breadcrumb)
    const brandElements = screen.getAllByText("Signova AI");
    expect(brandElements.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("ISL Platform")).toBeDefined();

    const sidebar = screen.getByRole("complementary", { name: /Primary Navigation/i });

    // All sidebar navigation links exist
    expect(within(sidebar).getByRole("link", { name: /^Dashboard$/i })).toBeDefined();
    expect(within(sidebar).getByRole("link", { name: /^Live Recognition$/i })).toBeDefined();
    expect(within(sidebar).getByRole("link", { name: /^Vocabulary$/i })).toBeDefined();
    expect(within(sidebar).getByRole("link", { name: /^Testing & History$/i })).toBeDefined();
    expect(within(sidebar).getByRole("link", { name: /^Settings$/i })).toBeDefined();
    expect(within(sidebar).getByRole("link", { name: /^About$/i })).toBeDefined();

    // Default route redirects to dashboard
    expect(await screen.findByRole("heading", { name: "Welcome to Signova AI" })).toBeDefined();
    await waitFor(() => {
      expect(screen.getByText("API Connected")).toBeDefined();
    });
  });

  it("tests every navigation link and verifies page content updates", async () => {
    render(<App />);

    const sidebar = screen.getByRole("complementary", { name: /Primary Navigation/i });

    // Navigate to Vocabulary
    const vocabLink = within(sidebar).getByRole("link", { name: /^Vocabulary$/i });
    fireEvent.click(vocabLink);
    expect(await screen.findByRole("heading", { name: "Vocabulary" })).toBeDefined();
    expect(screen.getByText(/Explore supported Indian Sign Language vocabulary/i)).toBeDefined();

    // Navigate to Testing & History
    const historyLink = within(sidebar).getByRole("link", { name: /^Testing & History$/i });
    fireEvent.click(historyLink);
    expect(await screen.findByRole("heading", { name: "Testing & History" })).toBeDefined();

    // Navigate to Settings
    const settingsLink = within(sidebar).getByRole("link", { name: /^Settings$/i });
    fireEvent.click(settingsLink);
    expect(await screen.findByRole("heading", { name: "Settings" })).toBeDefined();

    // Navigate to About
    const aboutLink = within(sidebar).getByRole("link", { name: /^About$/i });
    fireEvent.click(aboutLink);
    expect(await screen.findByRole("heading", { name: "About Signova AI" })).toBeDefined();

    // Navigate to Live Recognition
    const recLink = within(sidebar).getByRole("link", { name: /^Live Recognition$/i });
    fireEvent.click(recLink);
    expect(
      await screen.findByText("Signova AI - Indian Sign Language Recognition")
    ).toBeDefined();

    // Navigate back to Dashboard
    const dashLink = within(sidebar).getByRole("link", { name: /^Dashboard$/i });
    fireEvent.click(dashLink);
    expect(await screen.findByRole("heading", { name: "Welcome to Signova AI" })).toBeDefined();
  });

  it("applies active navigation state on current route", async () => {
    render(<App />);

    const sidebar = screen.getByRole("complementary", { name: /Primary Navigation/i });
    const dashLink = within(sidebar).getByRole("link", { name: /^Dashboard$/i });
    const vocabLink = within(sidebar).getByRole("link", { name: /^Vocabulary$/i });

    // Initially Dashboard is active after initial redirect
    await waitFor(() => {
      expect(dashLink.className).toContain("active");
    });
    expect(vocabLink.className).not.toContain("active");

    // Click Vocabulary
    fireEvent.click(vocabLink);
    expect(vocabLink.className).toContain("active");
    expect(dashLink.className).not.toContain("active");

    await waitFor(() => {
      expect(screen.getByText("API Connected")).toBeDefined();
    });
  });

  it("toggles desktop sidebar collapsed state", async () => {
    render(<App />);

    const sidebar = screen.getByRole("complementary", { name: /Primary Navigation/i });
    expect(sidebar.className).not.toContain("collapsed");

    // Click desktop collapse button in footer
    const collapseBtn = screen.getByRole("button", { name: /Collapse sidebar/i });
    fireEvent.click(collapseBtn);
    expect(sidebar.className).toContain("collapsed");

    // Click expand button
    const expandBtn = screen.getByRole("button", { name: /Expand sidebar/i });
    fireEvent.click(expandBtn);
    expect(sidebar.className).not.toContain("collapsed");

    await waitFor(() => {
      expect(screen.getByText("API Connected")).toBeDefined();
    });
  });

  it("toggles mobile drawer navigation", async () => {
    // Simulate mobile viewport width
    window.innerWidth = 500;

    render(<App />);

    const sidebar = screen.getByRole("complementary", { name: /Primary Navigation/i });
    expect(sidebar.className).not.toContain("mobile-open");

    // Toggle via navbar button
    const toggleBtn = screen.getByRole("button", { name: /Toggle navigation menu/i });
    fireEvent.click(toggleBtn);
    expect(sidebar.className).toContain("mobile-open");

    // Click close drawer button in mobile header
    const closeBtn = screen.getByRole("button", { name: /Close navigation drawer/i });
    fireEvent.click(closeBtn);
    expect(sidebar.className).not.toContain("mobile-open");

    // Reopen and test close on backdrop click
    fireEvent.click(toggleBtn);
    expect(sidebar.className).toContain("mobile-open");
    const backdrop = screen.getByLabelText("Close mobile navigation");
    fireEvent.click(backdrop);
    expect(sidebar.className).not.toContain("mobile-open");

    await waitFor(() => {
      expect(screen.getByText("API Connected")).toBeDefined();
    });
  });

  it("displays backend connection status accurately", async () => {
    render(<App />);

    const badge = screen.getByTestId("backend-status-badge");
    expect(badge).toBeDefined();

    // When API is healthy
    await waitFor(() => {
      expect(screen.getByText("API Connected")).toBeDefined();
    });

    expect(badge.className).toContain("healthy");
  });

  it("shows offline status and retry button when backend is unreachable", async () => {
    vi.spyOn(api, "fetchHealth").mockRejectedValueOnce(new Error("Network Error"));

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("API Offline")).toBeDefined();
    });

    const retryBtn = screen.getByRole("button", { name: /Retry backend connection check/i });
    expect(retryBtn).toBeDefined();

    // Mock success on retry
    vi.spyOn(api, "fetchHealth").mockResolvedValueOnce({ status: "healthy" });
    fireEvent.click(retryBtn);

    await waitFor(() => {
      expect(screen.getByText("API Connected")).toBeDefined();
    });
  });
});
