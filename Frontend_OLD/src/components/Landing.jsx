import { useNavigate } from "react-router-dom";
import { useState } from "react";

export default function Landing() {
  const navigate = useNavigate();
  const [hovered, setHovered] = useState(null);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Background Layers */}
      <div 
        className="fixed inset-0 z-0"
        style={{
          backgroundImage: 'url(https://images.unsplash.com/photo-1764032757764-195da62d6472?w=1920)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
      />
      <div className="fixed inset-0 z-0 bg-black opacity-60" />
      <div className="fixed inset-0 z-0 bg-gradient-to-br from-purple-900 via-pink-900 to-violet-900 opacity-40" />

      {/* Header */}
      <header className="relative z-10 px-10 py-6 flex justify-between items-center border-b border-white/20 bg-purple-900/80 backdrop-blur-md">
        <div className="text-lg text-white/80 font-medium">
          AI Facial Proportion System
        </div>
        <h1 className="text-3xl font-bold text-white tracking-wide">Pro-Portion</h1>
      </header>

      {/* About Section */}
      <section className="relative z-10 py-16 text-center">
        <div className="max-w-3xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-white mb-4">About the System</h2>
          <p className="text-white/90 text-lg">
            Pro-Portion helps artists construct accurate facial structures using the classical Loomis method enhanced with artificial intelligence.
          </p>
        </div>
      </section>

      {/* Main Content */}
      <main className="relative z-10 flex-1 flex items-center justify-center px-8 pb-20">
        <div className="flex flex-col md:flex-row gap-10">
          
          {/* Static Button */}
          <div
            onMouseEnter={() => setHovered("static")}
            onMouseLeave={() => setHovered(null)}
            onClick={() => navigate("/static")}
            style={{
              width: hovered === "static" ? "380px" : "260px",
              height: hovered === "static" ? "420px" : "120px",
            }}
            className="relative cursor-pointer transition-all duration-500 bg-white/10 backdrop-blur-xl border border-white/30 rounded-3xl shadow-2xl overflow-hidden"
          >
            {/* Collapsed */}
            <div
              style={{ opacity: hovered === "static" ? 0 : 1 }}
              className="absolute inset-0 flex items-center justify-center gap-3 transition-opacity duration-300"
            >
              <span className="text-3xl">📸</span>
              <span className="text-xl font-semibold text-white">Static</span>
            </div>

            {/* Expanded */}
            <div
              style={{ opacity: hovered === "static" ? 1 : 0 }}
              className="absolute inset-0 p-6 flex flex-col justify-between transition-opacity duration-300"
            >
              <div>
                <div className="text-5xl text-center mb-3">📸</div>
                <h3 className="text-2xl font-bold text-white text-center mb-2">Static Analysis</h3>
                <p className="text-sm text-white/80 text-center">
                  Upload images and receive proportion measurements with Loomis construction guidance.
                </p>
              </div>
              <button className="w-full bg-gradient-to-r from-pink-500 to-rose-500 text-white py-3 rounded-full font-semibold hover:scale-105 transition shadow-lg">
                Proceed →
              </button>
            </div>
          </div>

          {/* Realtime Button */}
          <div
            onMouseEnter={() => setHovered("realtime")}
            onMouseLeave={() => setHovered(null)}
            onClick={() => navigate("/realtime")}
            style={{
              width: hovered === "realtime" ? "380px" : "260px",
              height: hovered === "realtime" ? "420px" : "120px",
            }}
            className="relative cursor-pointer transition-all duration-500 bg-white/10 backdrop-blur-xl border border-white/30 rounded-3xl shadow-2xl overflow-hidden"
          >
            {/* Collapsed */}
            <div
              style={{ opacity: hovered === "realtime" ? 0 : 1 }}
              className="absolute inset-0 flex items-center justify-center gap-3 transition-opacity duration-300"
            >
              <span className="text-3xl">📹</span>
              <span className="text-xl font-semibold text-white">Realtime</span>
            </div>

            {/* Expanded */}
            <div
              style={{ opacity: hovered === "realtime" ? 1 : 0 }}
              className="absolute inset-0 p-6 flex flex-col justify-between transition-opacity duration-300"
            >
              <div>
                <div className="text-5xl text-center mb-3">📹</div>
                <h3 className="text-2xl font-bold text-white text-center mb-2">Live Assistant</h3>
                <p className="text-sm text-white/80 text-center">
                  Use your webcam for real-time head pose tracking and adaptive grid alignment.
                </p>
              </div>
              <button className="w-full bg-gradient-to-r from-purple-500 to-violet-500 text-white py-3 rounded-full font-semibold hover:scale-105 transition shadow-lg">
                Proceed →
              </button>
            </div>
          </div>

        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/20 py-6 text-center text-white/80 bg-purple-900/80 backdrop-blur-md">
        AI-powered facial proportion analysis using the classical Loomis method for artists and designers.
      </footer>
    </div>
  );
}
