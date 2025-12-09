import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { RefreshCw, Image, Users, Activity, Loader2, Camera } from 'lucide-react'
import './PhotographerView.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function PhotographerView() {
  const [stats, setStats] = useState(null)
  const [feed, setFeed] = useState([])
  const [loading, setLoading] = useState(false)
  const [page, setPage] = useState(1)

  useEffect(() => {
    loadStats()
    loadFeed()
  }, [])

  const loadStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/stats`)
      setStats(response.data)
    } catch (err) {
      console.error('Failed to load stats:', err)
    }
  }

  const loadFeed = async (pageNum = 1) => {
    setLoading(true)
    try {
      const response = await axios.get(`${API_URL}/feed`, {
        params: { page: pageNum, page_size: 20 }
      })
      if (pageNum === 1) {
        setFeed(response.data)
      } else {
        setFeed([...feed, ...response.data])
      }
    } catch (err) {
      console.error('Failed to load feed:', err)
    } finally {
      setLoading(false)
    }
  }

  const loadMore = () => {
    const nextPage = page + 1
    setPage(nextPage)
    loadFeed(nextPage)
  }

  return (
    <div className="photographer-view">
      <div className="dashboard-header">
        <h2>Dashboard</h2>
        <button onClick={loadStats} className="refresh-button">
          <RefreshCw size={16} />
          <span>Refresh</span>
        </button>
      </div>

      {stats && (
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon">
              <Image size={24} />
            </div>
            <div className="stat-value">{stats.images_processed || 0}</div>
            <div className="stat-label">Images Processed</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">
              <Users size={24} />
            </div>
            <div className="stat-value">{stats.unique_faces || 0}</div>
            <div className="stat-label">Unique Faces Detected</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">
              <Activity size={24} />
            </div>
            <div className="stat-value">{stats.status || 'Active'}</div>
            <div className="stat-label">System Status</div>
          </div>
        </div>
      )}

      <div className="qr-section">
        <h3>Share with Guests</h3>
        <div className="qr-info">
          <p>Guests can access photos at:</p>
          <code className="url-display">http://photos.local</code>
          <p className="qr-note">(or scan QR code when mDNS is configured)</p>
        </div>
      </div>

      <div className="live-feed-section">
        <h3>Live Feed</h3>
        {feed.length > 0 ? (
          <>
            <div className="feed-grid">
              {feed.map((photo) => (
                <div key={photo.photo_id} className="feed-photo-card">
                  <img
                    src={`${API_URL}${photo.thumbnail_url}`}
                    alt={photo.caption || 'Photo'}
                    className="feed-thumbnail"
                  />
                  {photo.caption && (
                    <div className="feed-caption">{photo.caption}</div>
                  )}
                  {photo.face_count !== undefined && (
                    <div className="feed-meta">
                      {photo.face_count} face{photo.face_count !== 1 ? 's' : ''}
                    </div>
                  )}
                </div>
              ))}
            </div>
            <button
              onClick={loadMore}
              disabled={loading}
              className="load-more-button"
            >
              {loading ? (
                <>
                  <Loader2 className="icon-spin" size={18} />
                  <span>Loading...</span>
                </>
              ) : (
                <>
                  <Camera size={18} />
                  <span>Load More</span>
                </>
              )}
            </button>
          </>
        ) : (
          <div className="empty-feed">
            <p>No photos processed yet. Add images to /images/raw directory.</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default PhotographerView

