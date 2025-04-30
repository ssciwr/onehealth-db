-- --------------------------------------------------------------------
-- ------      POSTGRESQL INITIAL SETUP for OMNIPATH SANDBOX      -----
-- --------------------------------------------------------------------
-- Authors:
--	Omnipath Team
--	Scientific Software Center
-- Last update: 02.04.2025

-- Conventions:
--      Database: 'django_metadata_db'
--          Purpose: To store all data related to the internal workings and operations of the Django backend
--          User/Role: postgres (root)
--          Tables to store:
--            * 'auth_group'
--            * 'auth_group_permissions'
--            * 'auth_permission'
--            * 'auth_user'
--            * 'auth_user_groups'
--            * 'django_admin_log'
--            * 'django_content_type'
--            * 'django_migrations'
--            * 'django_session'
--      Database: 'omnipath_db_sandbox'
--          Purpose: To store test data for validating and simulating Omnipath features.
--          User/Role: omnipathuser
--          Tables to store:
--            * omnipath_annotations
--            * omnipath_complexes
--            * omnipath_enzptm
--            * omnipath_interactions
--            * omnipath_intercell

-- WARNING: RUN WITH POSTGRES ROOT USER ('postgres')

-- Query 1. Create 'django_metadata_db' database
CREATE DATABASE django_metadata_db;

-- Query 2. Create 'omnipath_db_sandbox' database
CREATE DATABASE omnipath_db_sandbox;

-- Query 3. Set permissions and role for 'omnipath_db_sandbox' database
CREATE USER omnipathuser WITH PASSWORD 'omnipath123';
ALTER DATABASE omnipath_db_sandbox OWNER TO omnipathuser;
GRANT ALL PRIVILEGES ON DATABASE omnipath_db_sandbox TO omnipathuser;
-- GRANT ALL ON SCHEMA public TO omnipathuser;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO omnipathuser;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO omnipathuser;
-- GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO omnipathuser;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO omnipathuser;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO omnipathuser;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO omnipathuser;


-- -------  Examples for accesing the databases from a terminal
-- Connection with root (postgres)
--$ sudo -u postgres psql

-- Connection with 'omnipathuser'
--$ psql -h localhost -p 5432 -d omnipath_db_sandbox -U omnipathuser
