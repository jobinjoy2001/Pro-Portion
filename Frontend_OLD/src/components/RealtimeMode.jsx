import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

export default function RealtimeMode() {
  const navigate = useNavigate()
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const [pose, setPose] = useState({ pitch: 0, yaw: 0, roll: 0 })
  const [viewType, setViewType] = useState('Front View')
  const [fps, setFps] = useState(0)
  const [connected, setConnected] = useState(false)
  const frameCountRef = useRef(0)
  const lastFpsUpdateRef = useRef(Date.now())

  useEffect(() => {
    startWebcam()
    return () => {
      // Cleanup
      if (wsRef.current) wsRef.current.close()
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          if (canvasRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth
            canvasRef.current.height = videoRef.current.videoHeight
          }
          connectWebSocket()
        }
      }
    } catch (err) {
      alert('Camera access denied: ' + err.message)
    }
  }

  const connectWebSocket = () => {
    wsRef.current = new WebSocket('ws://localhost:8000/ws/realtime-grid')
    
    wsRef.current.onopen = () => {
      console.log('WebSocket connected')
      setConnected(true)
      processFrame()
    }
    
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.status === 'success') {
        drawGrid(data.grid, data.pose, data.view_type)
        setPose(data.pose)
        setViewType(data.view_type)
      } else {
        clearCanvas()
      }
      
      updateFPS()
    }
    
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error)
      setConnected(false)
    }
    
    wsRef.current.onclose = () => {
      console.log('WebSocket closed')
      setConnected(false)
    }
  }

  const processFrame = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && videoRef.current) {
      const canvas = document.createElement('canvas')
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(videoRef.current, 0, 0)
      
      canvas.toBlob(blob => {
        if (blob && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          blob.arrayBuffer().then(buffer => {
            wsRef.current.send(buffer)
          })
        }
      }, 'image/jpeg', 0.8)
    }
    
    requestAnimationFrame(processFrame)
  }

  const drawGrid = (grid, pose, viewType) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    ctx.strokeStyle = '#00ff00'
    ctx.lineWidth = 3
    ctx.font = 'bold 16px Arial'
    ctx.fillStyle = '#00ff00'
    ctx.shadowColor = '#000'
    ctx.shadowBlur = 5
    
    // Draw bounding box
    const box = grid.bounding_box
    ctx.strokeRect(box.left, box.top, box.right - box.left, box.bottom - box.top)
    
    // Draw vertical center line
    ctx.beginPath()
    ctx.moveTo(grid.vertical_center.x, grid.vertical_center.y1)
    ctx.lineTo(grid.vertical_center.x, grid.vertical_center.y2)
    ctx.stroke()
    
    // Draw horizontal lines
    grid.horizontal_lines.forEach(line => {
      ctx.beginPath()
      ctx.moveTo(line.x1, line.y)
      ctx.lineTo(line.x2, line.y)
      ctx.stroke()
      ctx.fillText(line.label, line.x2 + 10, line.y + 5)
    })
    
    // Draw eye line
    ctx.strokeStyle = '#ffff00'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(grid.eye_line.x1, grid.eye_line.y)
    ctx.lineTo(grid.eye_line.x2, grid.eye_line.y)
    ctx.stroke()
  }

  const clearCanvas = () => {
    const canvas = canvasRef.current
    if (canvas) {
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
  }

  const updateFPS = () => {
    frameCountRef.current++
    const now = Date.now()
    if (now - lastFpsUpdateRef.current >= 1000) {
      setFps(frameCountRef.current)
      frameCountRef.current = 0
      lastFpsUpdateRef.current = now
    }
  }

  const getViewTypeColor = (viewType) => {
    if (viewType.includes('Front')) return 'bg-green-500'
    if (viewType.includes('3/4')) return 'bg-yellow-500'
    if (viewType.includes('Profile')) return 'bg-orange-500'
    if (viewType.includes('Tilted')) return 'bg-purple-500'
    return 'bg-cyan-500'
  }

  return (
    <div className="relative w-screen h-screen bg-black overflow-hidden">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="absolute w-full h-full object-cover"
      />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full"
      />
      
      {/* Info Panel */}
      <div className="absolute top-5 left-5 bg-black/80 text-green-400 p-5 rounded-xl border-2 border-green-400 min-w-[300px] font-mono">
        <div className="text-center text-xl text-cyan-400 mb-4">📹 LIVE 3D GRID</div>
        <div className="space-y-2">
          <div className="flex justify-between border-b border-green-400/30 pb-2">
            <span className="text-cyan-400">Yaw (Turn):</span>
            <span className="font-bold">{pose.yaw}°</span>
          </div>
          <div className="flex justify-between border-b border-green-400/30 pb-2">
            <span className="text-cyan-400">Pitch (Tilt):</span>
            <span className="font-bold">{pose.pitch}°</span>
          </div>
          <div className="flex justify-between border-b border-green-400/30 pb-2">
            <span className="text-cyan-400">Roll (Lean):</span>
            <span className="font-bold">{pose.roll}°</span>
          </div>
          <div className="flex justify-between border-b border-green-400/30 pb-2">
            <span className="text-cyan-400">FPS:</span>
            <span className="font-bold">{fps}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-cyan-400">Status:</span>
            <span className={`font-bold ${connected ? 'text-green-400' : 'text-red-400'}`}>
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      {/* View Type Display */}
      <div className={`absolute bottom-10 left-1/2 -translate-x-1/2 ${getViewTypeColor(viewType)} text-black px-10 py-4 rounded-full font-bold text-2xl shadow-2xl`}>
        {viewType}
      </div>

      {/* Controls */}
      <div className="absolute top-5 right-5 flex gap-3">
        <button
          onClick={() => navigate('/static')}
          className="bg-green-500/80 hover:bg-green-500 text-black px-5 py-2 rounded-lg font-bold transition"
        >
          📸 Static Mode
        </button>
        <button
          onClick={() => navigate('/')}
          className="bg-green-500/80 hover:bg-green-500 text-black px-5 py-2 rounded-lg font-bold transition"
        >
          ← Back
        </button>
      </div>

      {/* Connection Status */}
      {connected && (
        <div className="absolute bottom-5 right-5 bg-green-500/80 text-black px-5 py-2 rounded-lg font-bold">
          ● CONNECTED
        </div>
      )}
    </div>
  )
}
