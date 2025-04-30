-- --------------------------------------------------------------------
-- ------      POSTGRESQL INITIAL SETUP for OMNIPATH SANDBOX      -----
-- --------------------------------------------------------------------
-- Authors:
--	Omnipath Team
--	Scientific Software Center
-- Last update: 02.04.2025

-- Conventions:
--      Database: 'onehealth_db'
--          Purpose: To store all data related to the internal workings and operations of the Django backend
--          User/Role: postgres (root)
--          Tables to store:
--            * ''



-- WARNING: RUN WITH POSTGRES ROOT USER ('postgres')

-- Query 1. Create 'onehealth_db' database
CREATE DATABASE onehealth_db;

-- Query 2. Set permissions and role for 'omnipath_db_sandbox' database
CREATE USER onehealth_admin WITH PASSWORD 'onehealth123';
ALTER DATABASE onehealth_db OWNER TO onehealth_admin;
GRANT ALL PRIVILEGES ON DATABASE onehealth_db TO onehealth_admin;

-- Query 3. Create the tables for the database here

-- -------  Examples for accesing the databases from a terminal
-- Connection with root (postgres)
--$ sudo -u postgres psql

-- Connection with 'omnipathuser'
--$ psql -h localhost -p 5432 -d onehealth_db -U onehealth_admin
