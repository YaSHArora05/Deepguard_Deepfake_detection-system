import type { PredictionResult } from "../services/api";

interface ResultCardProps {
  result: PredictionResult;
}

function ResultCard({
  result,
}: ResultCardProps) {
  const prediction =
    result.prediction?.toLowerCase() || "unavailable";

  const isReal = prediction === "real";
  const isFake = prediction === "fake";

  function formatPercentage(
    value?: number
  ) {
    if (value === undefined) {
      return "Not available";
    }

    const percentage =
      value <= 1 ? value * 100 : value;

    return `${percentage.toFixed(2)}%`;
  }

  function getPercentageValue(
    value?: number
  ) {
    if (value === undefined) {
      return 0;
    }

    return value <= 1 ? value * 100 : value;
  }

  function getModelName(fileType: string) {
    if (fileType === "image") {
      return "ResNet18 CNN";
    }

    if (fileType === "video") {
      return "CNN + LSTM";
    }

    if (fileType === "audio") {
      return "Audio CNN + LSTM";
    }

    return "Deep Learning Model";
  }

  if (prediction === "unavailable") {
    return (
      <section className="result-card unavailable-result">
        <h2>Model unavailable</h2>

        <p>
          {result.message ||
            "The selected model has not been trained yet."}
        </p>
      </section>
    );
  }

  const confidenceValue =
    getPercentageValue(result.confidence);

  const fakeValue =
    getPercentageValue(
      result.fake_probability
    );

  const realValue =
    getPercentageValue(
      result.real_probability
    );

  return (
    <section className="result-card">
      <div className="result-header">
        <div>
          <p className="section-label">
            ANALYSIS COMPLETE
          </p>

          <h2>Detection Result</h2>
        </div>

        <span
          className={`status-badge ${
            isReal
              ? "real-badge"
              : isFake
                ? "fake-badge"
                : ""
          }`}
        >
          {prediction.toUpperCase()}
        </span>
      </div>

      <div
        className={`prediction-banner ${
          isReal
            ? "real-result"
            : "fake-result"
        }`}
      >
        <span className="prediction-symbol">
          {isReal ? "✓" : "!"}
        </span>

        <div>
          <span>Prediction</span>
          <strong>
            {prediction.toUpperCase()}
          </strong>
        </div>
      </div>

      <div className="confidence-section">
        <div className="metric-heading">
          <span>Confidence score</span>

          <strong>
            {formatPercentage(
              result.confidence
            )}
          </strong>
        </div>

        <div className="progress-track">
          <div
            className={`progress-fill ${
              isReal
                ? "real-progress"
                : "fake-progress"
            }`}
            style={{
              width: `${Math.min(
                confidenceValue,
                100
              )}%`,
            }}
          />
        </div>
      </div>

      <div className="probability-grid">
        <div className="probability-card">
          <span>Real probability</span>

          <strong>
            {formatPercentage(
              result.real_probability
            )}
          </strong>

          <div className="small-progress-track">
            <div
              className="small-progress-fill real-progress"
              style={{
                width: `${Math.min(
                  realValue,
                  100
                )}%`,
              }}
            />
          </div>
        </div>

        <div className="probability-card">
          <span>Fake probability</span>

          <strong>
            {formatPercentage(
              result.fake_probability
            )}
          </strong>

          <div className="small-progress-track">
            <div
              className="small-progress-fill fake-progress"
              style={{
                width: `${Math.min(
                  fakeValue,
                  100
                )}%`,
              }}
            />
          </div>
        </div>
      </div>

      <div className="result-details">
        <div>
          <span>File</span>
          <strong>{result.filename}</strong>
        </div>

        <div>
          <span>Detected type</span>
          <strong>
            {result.file_type?.toUpperCase()}
          </strong>
        </div>

        <div>
          <span>Model used</span>
          <strong>
            {getModelName(result.file_type)}
          </strong>
        </div>
      </div>

      {result.message && (
        <p className="result-message">
          {result.message}
        </p>
      )}
    </section>
  );
}

export default ResultCard;