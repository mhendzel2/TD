## Simplification Tasks for Personal Use

### Phase 1: Update todo.md with simplification tasks
- [x] Add simplification tasks to todo.md

### Phase 2: Implement Docker Compose and related script simplifications
- [x] Simplify docker-compose.yml (remove prod, nginx-lb)
- [x] Adjust frontend service to expose port directly
- [x] Simplify deploy.sh (remove complex env setup, secret generation, db migrations)
- [x] Simplify monitor.sh (basic container health check)

### Phase 3: Implement database and caching simplifications
- [x] Migrate database from PostgreSQL to SQLite
- [x] Remove Redis caching and implement in-memory/file-based caching

### Phase 4: Implement task queue, notification, and security simplifications
- [x] Remove Celery task queue and handle tasks synchronously or with basic async
- [x] Remove email notification functionality
- [x] Relax advanced security features (rate limiting, strict security headers)

### Phase 5: Implement logging, monitoring, and deployment script simplifications
- [x] Reduce logging verbosity and remove external logging integrations
- [x] Simplify monitor.sh further if needed

### Phase 6: Update documentation and create final deliverable
- [ ] Update README.md to reflect simplifications
- [ ] Update other relevant documentation (e.g., API.md, DEPLOYMENT.md)
- [ ] Create new final deliverable package


