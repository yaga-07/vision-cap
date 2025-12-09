import React, { useState, useRef, useCallback } from 'react'
import Webcam from 'react-webcam'
import axios from 'axios'
import { Search, Camera, Download, CheckCircle2, Loader2, Sparkles, Image as ImageIcon } from 'lucide-react'
import './GuestView.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function GuestView() {
  const [searchMode, setSearchMode] = useState('text') // 'text' or 'face'
  const [textQuery, setTextQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const webcamRef = useRef(null)

  const searchByText = async () => {
    if (!textQuery.trim()) return
    
    setLoading(true)
    setError(null)
    
    try {
      const response = await axios.get(`${API_URL}/search/text`, {
        params: { query: textQuery, limit: 20 }
      })
      setResults(response.data)
    } catch (err) {
      setError('Search failed. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const captureAndSearch = useCallback(async () => {
    const imageSrc = webcamRef.current?.getScreenshot()
    if (!imageSrc) return

    setLoading(true)
    setError(null)

    try {
      // Convert base64 to blob
      const response = await fetch(imageSrc)
      const blob = await response.blob()
      
      const formData = new FormData()
      formData.append('file', blob, 'selfie.jpg')

      const searchResponse = await axios.post(`${API_URL}/search/face`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      
      setResults(searchResponse.data)
    } catch (err) {
      setError('Face search failed. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [])

  const searchSimilarImages = async (photoId) => {
    if (!photoId) return
    
    setLoading(true)
    setError(null)
    
    try {
      const response = await axios.get(`${API_URL}/search/similar`, {
        params: { photo_id: photoId, limit: 20 }
      })
      setResults(response.data)
    } catch (err) {
      setError('Failed to find similar images. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const downloadImage = async (imageUrl) => {
    try {
      const response = await fetch(`${API_URL}${imageUrl}`)
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `photo-${Date.now()}.jpg`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      console.error('Download failed:', err)
    }
  }

  return (
    <div className="guest-view">
      <div className="guest-hero">
        <div className="hero-content">
          <div className="hero-icon">
            <Sparkles size={32} />
          </div>
          <h1 className="hero-title">Find Your Photos</h1>
          <p className="hero-subtitle">Search by description or upload your photo to find yourself</p>
        </div>
      </div>

      <div className="search-container">
        <div className="search-mode-toggle">
          <button
            className={searchMode === 'text' ? 'active' : ''}
            onClick={() => setSearchMode('text')}
          >
            <Search size={18} />
            <span>Text Search</span>
          </button>
          <button
            className={searchMode === 'face' ? 'active' : ''}
            onClick={() => setSearchMode('face')}
          >
            <Camera size={18} />
            <span>Face Search</span>
          </button>
        </div>

        {searchMode === 'text' ? (
          <div className="text-search">
            <input
              type="text"
              placeholder="Search for photos... (e.g., 'dancing', 'red dress', 'cake')"
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && searchByText()}
              className="search-input"
            />
            <button onClick={searchByText} className="search-button" disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="icon-spin" size={18} />
                  <span>Searching...</span>
                </>
              ) : (
                <>
                  <Search size={18} />
                  <span>Search</span>
                </>
              )}
            </button>
          </div>
        ) : (
          <div className="face-search">
            <div className="webcam-container">
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className="webcam"
              />
            </div>
            <button onClick={captureAndSearch} className="capture-button" disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="icon-spin" size={18} />
                  <span>Searching...</span>
                </>
              ) : (
                <>
                  <Camera size={18} />
                  <span>Find My Photos</span>
                </>
              )}
            </button>
          </div>
        )}

        {error && <div className="error-message">{error}</div>}
      </div>

      {results.length > 0 && (
        <div className="results-section">
          <h2>Found {results.length} photo(s)</h2>
          <div className="results-grid">
            {results.map((photo) => (
              <div key={photo.photo_id} className="photo-card">
                <img
                  src={`${API_URL}${photo.thumbnail_url}`}
                  alt={photo.caption || 'Photo'}
                  className="photo-thumbnail"
                  onClick={() => searchSimilarImages(photo.photo_id)}
                  style={{ cursor: 'pointer' }}
                  title="Click to find similar images"
                  onLongPress={() => downloadImage(photo.image_url)}
                />
                {photo.caption && (
                  <div className="photo-caption">{photo.caption}</div>
                )}
                {photo.similarity_score && (
                  <div className="similarity-score">
                    <CheckCircle2 size={14} />
                    <span>Match: {(photo.similarity_score * 100).toFixed(1)}%</span>
                  </div>
                )}
                <div className="photo-actions">
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      searchSimilarImages(photo.photo_id)
                    }}
                    className="similar-button"
                    title="Find similar images"
                  >
                    <ImageIcon size={16} />
                    <span>Find Similar</span>
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      downloadImage(photo.image_url)
                    }}
                    className="download-button"
                  >
                    <Download size={16} />
                    <span>Download</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {results.length === 0 && !loading && (
        <div className="empty-state">
          <div className="empty-icon">
            <ImageIcon size={64} />
          </div>
          <h3>Start Your Search</h3>
          <p>Use text search to find photos by description, or use face search to find photos of yourself</p>
        </div>
      )}
    </div>
  )
}

export default GuestView

