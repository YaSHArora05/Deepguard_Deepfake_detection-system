import type {
  ChangeEvent,
  DragEvent,
} from "react";

interface FileUploadProps {
  selectedFile: File | null;
  loading: boolean;
  onFileSelect: (file: File) => void;
  onAnalyze: () => void;
}

function FileUpload({
  selectedFile,
  loading,
  onFileSelect,
  onAnalyze,
}: FileUploadProps) {
  function validateAndSelect(file: File) {
    const allowedTypes = [
      "image/",
      "video/",
      "audio/",
    ];

    const validFile = allowedTypes.some((type) =>
      file.type.startsWith(type)
    );

    if (!validFile) {
      alert("Please select an image, video, or audio file.");
      return;
    }

    onFileSelect(file);
  }

  function handleFileChange(
    event: ChangeEvent<HTMLInputElement>
  ) {
    const file = event.target.files?.[0];

    if (file) {
      validateAndSelect(file);
    }
  }

  function handleDrop(
    event: DragEvent<HTMLDivElement>
  ) {
    event.preventDefault();

    const file = event.dataTransfer.files?.[0];

    if (file) {
      validateAndSelect(file);
    }
  }

  function handleDragOver(
    event: DragEvent<HTMLDivElement>
  ) {
    event.preventDefault();
  }

  function formatFileSize(size: number) {
    if (size < 1024) {
      return `${size} bytes`;
    }

    if (size < 1024 * 1024) {
      return `${(size / 1024).toFixed(2)} KB`;
    }

    return `${(
      size /
      (1024 * 1024)
    ).toFixed(2)} MB`;
  }

  return (
    <section className="upload-section">
      <div className="section-heading">
        <p className="section-label">
          MULTIMODAL ANALYSIS
        </p>

        <h2>Upload media for detection</h2>

        <p>
          Select an image, video, or audio file to
          analyze using the trained deep-learning
          models.
        </p>
      </div>

      <div
        className="upload-box"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <div className="upload-icon">↑</div>

        <h3>Drag and drop your file here</h3>

        <p>or choose a file from your computer</p>

        <label className="browse-button">
          Browse File

          <input
            type="file"
            accept="image/*,video/*,audio/*"
            onChange={handleFileChange}
          />
        </label>
      </div>

      {selectedFile && (
        <div className="selected-file-card">
          <div className="file-symbol">
            {selectedFile.type.startsWith("image/")
              ? "IMG"
              : selectedFile.type.startsWith("video/")
                ? "VID"
                : "AUD"}
          </div>

          <div className="file-information">
            <strong>{selectedFile.name}</strong>

            <span>
              {selectedFile.type || "Unknown type"}
            </span>

            <span>
              {formatFileSize(selectedFile.size)}
            </span>
          </div>

          <span className="file-ready">Ready</span>
        </div>
      )}

      <button
        className="analyze-button"
        onClick={onAnalyze}
        disabled={!selectedFile || loading}
      >
        {loading
          ? "Analyzing file..."
          : "Analyze File"}
      </button>

      {loading && (
        <div className="loading-panel">
          <div className="spinner" />

          <div>
            <strong>Running deepfake analysis</strong>
            <p>
              Preprocessing media and running the
              selected model.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}

export default FileUpload;