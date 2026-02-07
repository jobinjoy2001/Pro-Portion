import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

export default function StaticMode() {
  const navigate = useNavigate()
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [mode, setMode] = useState('standard')

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResult(null)
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select an image first!')
      return
    }

    setLoading(true)
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const endpoint = mode === 'tutorial' 
        ? 'http://localhost:8000/process-tutorial'
        : 'http://localhost:8000/process'
      
      console.log('Sending request to:', endpoint)
      
      const response = await axios.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      
      console.log('Response:', response.data)
      setResult(response.data)
    } catch (error) {
      console.error('Upload error:', error)
      alert('Error processing image: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  const downloadImage = (filename) => {
    window.open(`http://localhost:8000/download/${filename}`, '_blank')
  }

  const downloadTutorialImage = (filename) => {
    window.open(`http://localhost:8000/download-tutorial/${filename}`, '_blank')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-4xl font-bold text-white">📸 Static Image Analysis</h1>
          <div className="flex gap-4">
            <button
              onClick={() => navigate('/realtime')}
              className="bg-white/20 hover:bg-white/30 text-white px-6 py-2 rounded-full backdrop-blur-lg transition"
            >
              📹 Switch to Live Mode
            </button>
            <button
              onClick={() => navigate('/')}
              className="bg-white/20 hover:bg-white/30 text-white px-6 py-2 rounded-full backdrop-blur-lg transition"
            >
              ← Back to Home
            </button>
          </div>
        </div>

        {/* Mode Selection */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 mb-6">
          <h3 className="text-white font-bold mb-4">Select Analysis Mode:</h3>
          <div className="flex gap-4">
            <button
              onClick={() => setMode('standard')}
              className={`flex-1 py-4 rounded-xl font-bold transition ${
                mode === 'standard'
                  ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white scale-105'
                  : 'bg-white/20 text-white/70 hover:bg-white/30'
              }`}
            >
              📊 Standard Analysis
              <p className="text-sm font-normal mt-1">Single processed image with all measurements</p>
            </button>
            <button
              onClick={() => setMode('tutorial')}
              className={`flex-1 py-4 rounded-xl font-bold transition ${
                mode === 'tutorial'
                  ? 'bg-gradient-to-r from-pink-500 to-rose-500 text-white scale-105'
                  : 'bg-white/20 text-white/70 hover:bg-white/30'
              }`}
            >
              📚 Tutorial Mode
              <p className="text-sm font-normal mt-1">6-step Loomis grid construction guide</p>
            </button>
          </div>
        </div>

        {/* Upload Section */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 mb-6">
          <div className="border-4 border-dashed border-white/30 rounded-xl p-12 text-center hover:border-white/50 transition cursor-pointer">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
              id="fileInput"
            />
            <label htmlFor="fileInput" className="cursor-pointer">
              <div className="text-6xl mb-4">📁</div>
              <p className="text-white text-xl mb-2">Click to upload an image</p>
              <p className="text-white/70">Supports: JPG, PNG, JPEG</p>
            </label>
          </div>

          {/* Image Preview and Upload Button */}
          {previewUrl && (
            <div className="mt-8 space-y-4">
              <div className="bg-white/20 rounded-xl p-4">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="max-h-96 mx-auto rounded-lg shadow-2xl"
                />
              </div>
              
              {/* UPLOAD BUTTON - Now more visible! */}
              <button
                onClick={handleUpload}
                disabled={loading}
                className="w-full bg-gradient-to-r from-green-400 to-blue-500 text-white py-6 rounded-xl font-bold text-2xl hover:scale-105 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-2xl"
              >
                {loading ? (
                  <span>⏳ Processing... Please wait</span>
                ) : (
                  <span>🚀 Analyze Image ({mode === 'tutorial' ? 'Tutorial Mode' : 'Standard Mode'})</span>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Results Section */}
        {result && (
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8">
            <h2 className="text-3xl font-bold text-white mb-6">📊 Analysis Results</h2>

            {mode === 'standard' ? (
              <div>
                {/* Standard Results */}
                <div className="grid md:grid-cols-2 gap-6 mb-6">
                  <div className="bg-white/20 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-white mb-4">📐 Measurements</h3>
                    <div className="space-y-2 text-white">
                      {result.measurements && Object.entries(result.measurements).map(([key, value]) => (
                        <div key={key} className="flex justify-between border-b border-white/20 pb-2">
                          <span className="capitalize">{key.replace(/_/g, ' ')}:</span>
                          <span className="font-bold">{typeof value === 'number' ? value.toFixed(2) : value}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-white/20 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-white mb-4">🎯 Analysis</h3>
                    <div className="space-y-3 text-white">
                      <div className="flex justify-between border-b border-white/20 pb-2">
                        <span>Proportion Score:</span>
                        <span className="font-bold text-green-300">{result.proportion_score?.toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between border-b border-white/20 pb-2">
                        <span>Face Shape:</span>
                        <span className="font-bold">{result.face_shape || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between border-b border-white/20 pb-2">
                        <span>Symmetry:</span>
                        <span className="font-bold">{result.symmetry?.toFixed(1) || 'N/A'}%</span>
                      </div>
                    </div>
                  </div>
                </div>

                {result.processed_image && (
                  <div className="text-center">
                    <img
                      src={`http://localhost:8000/download/${result.processed_image}`}
                      alt="Processed"
                      className="max-w-full mx-auto rounded-lg shadow-2xl mb-4"
                    />
                    <button
                      onClick={() => downloadImage(result.processed_image)}
                      className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-8 py-3 rounded-full font-bold hover:scale-105 transition"
                    >
                      ⬇️ Download Result
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div>
                {/* Tutorial Results */}
                <div className="grid md:grid-cols-3 gap-4 mb-6">
                  {result.tutorial_steps && result.tutorial_steps.map((step, index) => (
                    <div key={index} className="bg-white/20 rounded-xl p-4">
                      <h4 className="text-white font-bold mb-2">{step.title}</h4>
                      <img
                        src={`http://localhost:8000/download-tutorial/${step.filename}`}
                        alt={step.title}
                        className="w-full rounded-lg mb-2"
                      />
                      <button
                        onClick={() => downloadTutorialImage(step.filename)}
                        className="w-full bg-blue-500 hover:bg-blue-600 text-white py-2 rounded-lg text-sm transition"
                      >
                        ⬇️ Download
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
