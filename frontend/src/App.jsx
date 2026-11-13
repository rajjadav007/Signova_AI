import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import MainLayout from "./components/layout/MainLayout";
import DashboardPage from "./pages/DashboardPage";
import LiveRecognitionPage from "./pages/LiveRecognitionPage";
import PlaceholderPage from "./components/common/PlaceholderPage";
import {
  BookOpen,
  History,
  Settings,
  Info,
} from "lucide-react";
import "./styles/globals.css";
import "./styles/layout.css";
import "./styles/dashboard.css";

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="recognition" element={<LiveRecognitionPage />} />
        <Route
          path="vocabulary"
          element={
            <PlaceholderPage
              title="Vocabulary"
              description="Explore supported Indian Sign Language vocabulary, gestures, and landmark references."
              icon={BookOpen}
            />
          }
        />
        <Route
          path="history"
          element={
            <PlaceholderPage
              title="Testing & History"
              description="Review recent gesture classification tests, accuracy benchmarks, and historical logs."
              icon={History}
            />
          }
        />
        <Route
          path="settings"
          element={
            <PlaceholderPage
              title="Settings"
              description="Configure camera inputs, recognition thresholds, API preferences, and speech options."
              icon={Settings}
            />
          }
        />
        <Route
          path="about"
          element={
            <PlaceholderPage
              title="About Signova AI"
              description="Signova AI is an accessible Indian Sign Language (ISL) recognition and translation platform powered by deep learning."
              icon={Info}
            />
          }
        />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
