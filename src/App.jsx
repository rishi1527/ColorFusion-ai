import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "https://colorfusion-api.onrender.com/api/colorize";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    setError("");
    setResult("");

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    if (selectedFile.size > 15 * 1024 * 1024) {
      setError("Image size must be under 15 MB.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

  const handleInputChange = (event) => {
    handleFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);
    handleFile(event.dataTransfer.files?.[0]);
  };

  const handleColorize = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult("");

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let message = "Colorization failed.";

        try {
          const data = await response.json();
          message = data.error || message;
        } catch {
          // Ignore non-JSON error responses.
        }

        throw new Error(message);
      }

      const blob = await response.blob();
      const resultUrl = URL.createObjectURL(blob);

      setResult(resultUrl);
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to the ColorFusion AI backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const chooseAnother = () => {
    if (preview) URL.revokeObjectURL(preview);
    if (result) URL.revokeObjectURL(result);

    setFile(null);
    setPreview("");
    setResult("");
    setError("");

    inputRef.current?.click();
  };

  const resetProject = () => {
    if (preview) URL.revokeObjectURL(preview);
    if (result) URL.revokeObjectURL(result);

    setFile(null);
    setPreview("");
    setResult("");
    setError("");
  };

  const formatSize = (bytes) => {
    if (!bytes) return "0 KB";

    const units = ["Bytes", "KB", "MB", "GB"];
    const index = Math.floor(Math.log(bytes) / Math.log(1024));

    return `${(bytes / Math.pow(1024, index)).toFixed(
      index === 0 ? 0 : 1
    )} ${units[index]}`;
  };

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <header className="topbar">
        <button className="brand" onClick={resetProject}>
          <span className="brand-mark">CF</span>

          <span className="brand-copy">
            <strong>ColorFusion</strong>
            <small>AI Photo Colorizer</small>
          </span>
        </button>

        <div className="status-pill">
          <span className="status-dot" />
          AI Ready
        </div>
      </header>

      <main className="main-content">
        {!file ? (
          <section className="hero-section">
            <div className="hero-badge">
              <span>✦</span>
              Powered by DDColor AI
            </div>

            <h1>
              Bring old photos
              <br />
              <span>back to life.</span>
            </h1>

            <p className="hero-description">
              Transform black & white memories into natural-looking
              color with AI-powered photo restoration.
            </p>

            <div
              className={`upload-card ${
                dragActive ? "drag-active" : ""
              }`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
            >
              <div className="upload-icon">
                <span>↑</span>
              </div>

              <h2>Drop your photo here</h2>

              <p>
                or <span>browse from your computer</span>
              </p>

              <div className="upload-meta">
                JPG · PNG · WEBP
                <span>•</span>
                Maximum 15 MB
              </div>

              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                onChange={handleInputChange}
                hidden
              />
            </div>

            {error && <div className="error-message">{error}</div>}
          </section>
        ) : (
          <section className="workspace">
            <div className="workspace-heading">
              <div>
                <div className="section-eyebrow">
                  PHOTO WORKSPACE
                </div>

                <h1>
                  {result ? "Your photo, reimagined." : "Ready to colorize."}
                </h1>

                <p>
                  {result
                    ? "Compare the original with your AI colorized result."
                    : "Your image is ready. Let ColorFusion do the magic."}
                </p>
              </div>

              <button
                className="ghost-button"
                onClick={chooseAnother}
                disabled={loading}
              >
                <span>＋</span>
                Choose another
              </button>
            </div>

            <div className={`photo-grid ${result ? "has-result" : ""}`}>
              <article className="photo-card">
                <div className="card-header">
                  <div>
                    <span className="card-label">ORIGINAL</span>
                    <h3>{file.name}</h3>
                  </div>

                  <span className="image-tag">B&W</span>
                </div>

                <div className="image-frame">
                  <img src={preview} alt="Original preview" />
                </div>

                <div className="file-info">
                  <span>{file.type.split("/")[1]?.toUpperCase() || "IMAGE"}</span>
                  <span>•</span>
                  <span>{formatSize(file.size)}</span>
                </div>
              </article>

              {result ? (
                <article className="photo-card result-card">
                  <div className="card-header">
                    <div>
                      <span className="card-label result-label">
                        COLORIZED RESULT
                      </span>
                      <h3>AI enhanced photo</h3>
                    </div>

                    <span className="image-tag success-tag">AI</span>
                  </div>

                  <div className="image-frame result-frame">
                    <img src={result} alt="Colorized result" />
                  </div>

                  <div className="result-actions">
                    <a
                      className="primary-button"
                      href={result}
                      download={`colorfusion-${file.name}`}
                    >
                      <span>↓</span>
                      Download Result
                    </a>

                    <button
                      className="secondary-button"
                      onClick={handleColorize}
                      disabled={loading}
                    >
                      Recolor
                    </button>
                  </div>
                </article>
              ) : (
                <article className="photo-card waiting-card">
                  <div className="waiting-content">
                    <div className="magic-orb">
                      <span>✦</span>
                    </div>

                    <span className="card-label">AI PREVIEW</span>

                    <h3>Ready when you are</h3>

                    <p>
                      Click below to transform your photo with
                      DDColor AI.
                    </p>

                    <button
                      className="primary-button colorize-button"
                      onClick={handleColorize}
                      disabled={loading}
                    >
                      {loading ? (
                        <>
                          <span className="spinner" />
                          Processing image...
                        </>
                      ) : (
                        <>
                          <span>✦</span>
                          Colorize Image
                        </>
                      )}
                    </button>

                    <small>
                      Processing happens locally through your
                      ColorFusion AI backend.
                    </small>
                  </div>
                </article>
              )}
            </div>

            {loading && (
              <div className="processing-bar">
                <div className="processing-spinner" />
                <div>
                  <strong>Colorizing your photo...</strong>
                  <span>DDColor AI is analyzing the image.</span>
                </div>
              </div>
            )}

            {error && <div className="error-message">{error}</div>}
          </section>
        )}
      </main>

      <footer className="footer">
        <span>ColorFusion</span>
        <span>AI-powered photo colorization</span>
      </footer>
    </div>
  );
}

export default App;
