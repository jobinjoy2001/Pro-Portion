import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Landing from './components/Landing'
import StaticMode from './components/StaticMode'
import RealtimeMode from './components/RealtimeMode'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/static" element={<StaticMode />} />
        <Route path="/realtime" element={<RealtimeMode />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
