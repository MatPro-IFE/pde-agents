-- PDE Simulation Database Initialization
-- Run automatically by Docker on first start

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Grant privileges (tables created by SQLAlchemy at app start)
GRANT ALL PRIVILEGES ON DATABASE pde_simulations TO pde_user;

-- Create MLflow schema (used by MLflow tracking server)
CREATE SCHEMA IF NOT EXISTS mlflow;
GRANT ALL PRIVILEGES ON SCHEMA mlflow TO pde_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA mlflow TO pde_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA mlflow GRANT ALL ON TABLES TO pde_user;
