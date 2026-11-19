import { useState } from "react";

import FileUpload from "./components/FileUpload";
import ResultCard from "./components/ResultCard";

import {
  predictFile,
  type PredictionResult,
} from "./services/api";

import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] =
    useState<File | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [result, setResult] =
    useState<PredictionResult | null>(null);

  const [error, setError] =
    useState("");

  function handleFileSelect(file: File) {
    setSelectedFile(file);
    setResult(null);
    setError("");
  }

  async function handleAnalyze() {
    if (!selectedFile) {
      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const predictionResult =
        await predictFile(selectedFile);

      setResult(predictionResult);
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : "Something went wrong"
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="application">
      <header className="navbar">
        <div className="brand">
          <div className="brand-icon">D</div>

          <div>
            <strong>DeepGuard</strong>
            <span>Deepfake Detection</span>
          </div>
        </div>

        <div className="navbar-status">
          <span className="status-dot" />
          Models online
        </div>
      </header>

      <main className="main-content">
        <section className="hero-section">
          <div className="hero-badge">
            AI-POWERED MEDIA VERIFICATION
          </div>

          <h1>
            Detect manipulated media with
            multimodal deep learning
          </h1>

          <p>
            Analyze images, videos, and audio using
            dedicated CNN and CNN-LSTM pipelines.
          </p>

          <div className="model-summary">
            <div>
              <span>IMAGE</span>
              <strong>ResNet18 CNN</strong>
            </div>

            <div>
              <span>VIDEO</span>
              <strong>CNN + LSTM</strong>
            </div>

            <div>
              <span>AUDIO</span>
              <strong>CNN + LSTM</strong>
            </div>
          </div>
        </section>

        <div className="workspace">
          <FileUpload
            selectedFile={selectedFile}
            loading={loading}
            onFileSelect={handleFileSelect}
            onAnalyze={handleAnalyze}
          />

          {error && (
            <div className="error-panel">
              <strong>Analysis failed</strong>
              <p>{error}</p>
            </div>
          )}

          {result && (
            <ResultCard result={result} />
          )}
        </div>

        <section className="information-section">
          <article>
            <span>01</span>
            <h3>Upload media</h3>
            <p>
              Select an image, video, or audio file
              from your device.
            </p>
          </article>

          <article>
            <span>02</span>
            <h3>AI analysis</h3>
            <p>
              The backend preprocesses the file and
              runs the matching trained model.
            </p>
          </article>

          <article>
            <span>03</span>
            <h3>Review result</h3>
            <p>
              View the prediction, confidence, and
              class probabilities.
            </p>
          </article>
        </section>
      </main>

      <footer>
        <p>
          DeepShield — Multimodal Deepfake Detection
          System
        </p>

        <p>
          React · FastAPI · PyTorch · OpenCV · Librosa
        </p>
      </footer>
    </div>
  );
}

export default App;