# Trading Dashboard Deployment Guide (Simplified for Personal Use)

This guide provides step-by-step instructions for deploying the Trading Dashboard for personal use.

## 🚀 Quick Deployment (Docker)

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM

### One-Click Deployment
```bash
git clone <repository-url>
cd trading-dashboard
./scripts/deploy.sh
```

Access your dashboard at: http://localhost

## 🔧 Development Environment

### Local Development Setup

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd trading-dashboard
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your settings
   python src/main.py
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   cp .env.example .env
   # Edit .env with your settings
   npm start
   ```

4. **ML Engine Setup**
   ```bash
   cd ml-engine
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python main.py
   ```

### Development URLs
- Frontend: http://localhost:3000
- Backend: http://localhost:3001
- ML Engine: http://localhost:8000

## 📊 Monitoring

### Health Checks

All services include health check endpoints:

```bash
curl http://localhost/health      # Frontend
curl http://localhost:3001/api/health  # Backend
curl http://localhost:8000/health      # ML Engine
```

### Monitoring Script

```bash
./scripts/monitor.sh
```

Provides basic container health status.

### Logging

Simplified logging to console for easier debugging.

## 🔄 Backup & Recovery

### Automated Backups

```bash
./scripts/backup.sh
```

Creates compressed backups of:
- SQLite database
- ML models
- Configuration files

### Restore Process

```bash
# Extract backup
tar xzf backup_file.tar.gz

# Restore database (example for SQLite)
# cp backup_database.db trading_dashboard.db

# Restore ML models
docker run --rm -v trading-dashboard_ml_models:/data -v $(pwd):/backup alpine tar xzf /backup/backup_ml_models.tar.gz -C /data
```

## 📞 Support

### Getting Help

**Documentation:**
- 📖 Main Documentation: [README.md](README.md)
- 🔧 API Reference: [API.md](API.md)
- 🚀 Deployment Guide: This document

**Community Support:**
- 📧 Email: support@trading-dashboard.com

---

**Deployment Guide Version: 1.0.0 (Simplified)**  
**Last Updated: June 15, 2025**


