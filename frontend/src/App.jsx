import React, { useState } from 'react'
import GuestView from './components/GuestView'
import PhotographerView from './components/PhotographerView'
import './App.css'

function App() {
  const [view, setView] = useState('guest') // 'guest' or 'photographer'

  return (
    <div className="App">
      <nav className="nav-bar">
        <h1>Vision Cap</h1>
        <div className="nav-buttons">
          <button 
            className={view === 'guest' ? 'active' : ''}
            onClick={() => setView('guest')}
          >
            <span>Guest View</span>
          </button>
          <button 
            className={view === 'photographer' ? 'active' : ''}
            onClick={() => setView('photographer')}
          >
            <span>Photographer</span>
          </button>
        </div>
      </nav>
      
      {view === 'guest' ? <GuestView /> : <PhotographerView />}
    </div>
  )
}

export default App

