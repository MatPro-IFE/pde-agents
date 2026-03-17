-- PDE Simulation Database Initialization
-- Run automatically by Docker on first start

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Grant privileges (tables created by SQLAlchemy at app start)
GRANT ALL PRIVILEGES ON DATABASE pde_simulations TO pde_user;

