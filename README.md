# Newsletter-Enhanced Trading Dashboard (Simplified for Personal Use)

A comprehensive trading intelligence system that combines newsletter analysis, machine learning predictions, and real-time market data to provide actionable trading insights.

## 🎯 Overview

The Newsletter-Enhanced Trading Dashboard is a sophisticated web application designed for individual traders who want to leverage multiple data sources for informed trading decisions. This simplified version focuses on ease of use and reduced resource consumption for personal deployment. It combines:

- **Newsletter Analysis**: AI-powered sentiment analysis of trading newsletters
- **Machine Learning Predictions**: Random Forest models for probability scoring
- **Real-time Updates**: WebSocket-based live data streaming
- **Portfolio Integration**: Position tracking and risk management
- **Multi-source Data**: Integration with market data providers

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   ML Engine     │
│   (React)       │◄──►│   (Flask)       │◄──►│   (FastAPI)     │
│   Port: 80      │    │   Port: 3001    │    │   Port: 8000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         │              ┌─────────────────┐
         └──────────────│     SQLite      │
                        └─────────────────┘
```

### Technology Stack

**Frontend:**
- React 18 with TypeScript
- Material-UI for components
- Socket.IO for real-time updates
- Recharts for data visualization

**Backend:**
- Flask with SQLAlchemy ORM (using SQLite)
- Flask-SocketIO for WebSocket support
- JWT authentication
- SQLite database

**ML Engine:**
- FastAPI framework
- scikit-learn for machine learning
- Advanced sentiment analysis
- Random Forest prediction models

**Infrastructure:**
- Docker containerization
- Automated deployment scripts

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Git
- 4GB+ RAM recommended

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd trading-dashboard
   ```

2. **Set up environment:**
   ```bash
   # A basic .env file will be created automatically on first run.
   # You can edit it if needed.
   ```

3. **Deploy with Docker:**
   ```bash
   ./scripts/deploy.sh
   ```

4. **Access the application:**
   - Frontend: http://localhost
   - Backend API: http://localhost:3001
   - ML Engine: http://localhost:8000

### Development Setup

For development with hot reloading:

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/main.py

# Frontend
cd frontend
npm install
npm start

# ML Engine
cd ml-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## 📊 Features

### Newsletter Analysis
- **Content Extraction**: Automatic parsing of HTML and text newsletters
- **Ticker Identification**: Advanced pattern matching with 200+ known tickers
- **Sentiment Analysis**: ML-powered sentiment scoring for each mentioned ticker
- **Priority Scoring**: Automated ranking based on sentiment strength and source credibility

### Machine Learning Pipeline
- **Random Forest Models**: Hierarchical prediction architecture
- **Feature Engineering**: 5 categories of features (sentiment, technical, fundamental, market, portfolio)
- **Confidence Intervals**: Statistical confidence bounds for predictions
- **Model Persistence**: Automatic saving and loading of trained models

### Real-time Updates
- **WebSocket Integration**: Live updates for predictions, portfolio, and alerts
- **User Rooms**: Targeted messaging to specific users
- **Channel Subscriptions**: Topic-based update channels
- **Automatic Reconnection**: Robust connection handling

### Portfolio Management
- **Position Tracking**: Real-time portfolio monitoring
- **Risk Metrics**: Greeks calculation and exposure analysis
- **Performance Analytics**: P&L tracking and performance metrics
- **Alert System**: Customizable notifications for trading opportunities

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Security
JWT_SECRET=your_jwt_secret_key

# External APIs
ALPHA_VANTAGE_API_KEY=your_api_key
POLYGON_API_KEY=your_api_key

# ML Engine
ML_ENGINE_URL=http://ml-engine:8000
MODEL_VERSION=1.0.0
```

### Service Configuration

Each service can be configured through environment variables:

- **Backend**: Database connections, JWT settings, ML engine URL
- **Frontend**: API endpoints, WebSocket URLs
- **ML Engine**: Model parameters

## 📚 API Documentation

### Authentication Endpoints

```
POST /api/auth/register    - User registration
POST /api/auth/login       - User login
POST /api/auth/refresh     - Token refresh
POST /api/auth/logout      - User logout
```

### Newsletter Endpoints

```
GET  /api/newsletters/           - Get user newsletters
POST /api/newsletters/analyze    - Analyze newsletter content
GET  /api/newsletters/{id}       - Get specific newsletter
GET  /api/newsletters/tickers/{ticker} - Get ticker mentions
```

### ML Endpoints

```
POST /api/ml/predict            - Get trading predictions
POST /api/ml/sentiment          - Analyze sentiment
GET  /api/ml/models/status      - Model status
POST /api/ml/models/train       - Train models
```

### Portfolio Endpoints

```
GET  /api/portfolio/positions   - Get current positions
POST /api/portfolio/positions   - Add new position
PUT  /api/portfolio/positions/{id} - Update position
DELETE /api/portfolio/positions/{id} - Remove position
```

### WebSocket Events

```
prediction_update    - New ML predictions
portfolio_update     - Portfolio changes
new_alert           - Trading alerts
market_update       - Market data updates
newsletter_update   - Newsletter analysis results
```

## 🧪 Testing

### Running Tests

```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm test

# ML Engine tests
cd ml-engine
python -m pytest tests/

# Integration tests
./scripts/test-integration.sh
```

### Test Coverage

- **Backend**: Unit tests for all API endpoints and services
- **Frontend**: Component tests and integration tests
- **ML Engine**: Model validation and prediction accuracy tests
- **End-to-end**: Full workflow testing with Docker

## 🔍 Monitoring

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

## 🔒 Security

### Authentication & Authorization
- JWT-based authentication with refresh tokens
- Secure password hashing with bcrypt

### Data Protection
- Basic input validation and sanitization

### Infrastructure Security
- Non-root containers for all services
- Network isolation with Docker networks
- Secrets management through environment variables

## 📈 Performance

### Optimization Features
- **Database**: SQLite for simplified local storage
- **Caching**: In-memory caching for frequently accessed data
- **Frontend**: Code splitting and lazy loading

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

## 🚀 Deployment

### Local Deployment

```bash
./scripts/deploy.sh
```

This will build and start all services using Docker Compose.

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### Code Standards

- **Python**: PEP 8 with Black formatting
- **JavaScript**: ESLint with Prettier
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: Minimum 80% code coverage

### Issue Reporting

Please use GitHub Issues for:
- Bug reports with reproduction steps
- Feature requests with use cases
- Documentation improvements
- Performance issues

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **scikit-learn** for machine learning capabilities
- **Flask** and **React** communities for excellent frameworks
- **Docker** for containerization technology
- **SQLite** for robust local data storage

## 📞 Support

For support and questions:
- 📧 Email: support@trading-dashboard.com
- 💬 Discord: [Trading Dashboard Community](https://discord.gg/trading-dashboard)
- 📖 Documentation: [docs.trading-dashboard.com](https://docs.trading-dashboard.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/trading-dashboard/issues)

---

**Built with ❤️ for the trading community**


