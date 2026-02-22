from __future__ import annotations

"""SQLite database utilities for the civic complaint system.

Tables (created on demand):

* users(id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT UNIQUE, phone TEXT,
        role TEXT, created_at TEXT)
* departments(id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT, category TEXT, officials TEXT)
* complaints(id TEXT PRIMARY KEY,
             user_id INTEGER,
             text TEXT,
             category TEXT,
             urgency TEXT,
             language TEXT,
             location TEXT,
             status TEXT,

The schema is intentionally simple and uses TEXT-encoded JSON for
probability vectors and SHAP summaries.
"""

import bcrypt
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DB_RELATIVE = Path("data") / "civic_complaints.db"
ENV_DB_PATH = "CIVIC_DB_PATH"

# Database configuration
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'civic_complaints.db')
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'database_schema.sql')


def _resolve_db_path(project_root: Optional[Path] = None) -> Path:
    """Resolve the SQLite database path.

    Order of precedence:
      1. ``CIVIC_DB_PATH`` environment variable (absolute or relative).
      2. ``project_root / data/civic_system.db``.

    Args:
        project_root: Optional project root path.

    Returns:
        Absolute path to the SQLite database file.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    env_path = os.getenv(ENV_DB_PATH)
    if env_path:
        db_path = Path(env_path)
        if not db_path.is_absolute():
            db_path = project_root / db_path
    else:
        db_path = project_root / DEFAULT_DB_RELATIVE

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@contextmanager
def get_connection(project_root: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding a SQLite connection.

    The connection uses ``row_factory=sqlite3.Row`` and commits on
    successful exit, rolling back on exceptions.
    """
    db_path = _resolve_db_path(project_root)
    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
    conn.execute("PRAGMA journal_mode = WAL")  # Use WAL mode for better concurrency
    conn.execute("PRAGMA synchronous = NORMAL")  # Balance between performance and safety
    conn.execute("PRAGMA cache_size = 2000")  # Increase cache size
    conn.execute("PRAGMA temp_store = MEMORY")  # Use memory for temporary tables
    conn.execute("PRAGMA busy_timeout = 30000")  # 30 second busy timeout
    try:
        yield conn
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        logger.exception("Database error: %s", exc)
        raise
    finally:
        conn.close()


def _ensure_default_admin(cur: sqlite3.Cursor) -> None:
    """Ensure a default admin account exists.

    The account uses the credentials specified in the project brief:
    ``admin@civiccomplaints.gov`` / ``Admin@123``. The password is stored
    as a bcrypt hash. In a real deployment this password must be changed
    on first login.
    """
    email = "admin@civiccomplaints.gov"
    cur.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cur.fetchone() is not None:
        return

    now = datetime.utcnow().isoformat()
    password_hash = bcrypt.hashpw("Admin@123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    cur.execute(
        """
        INSERT INTO users(name, email, phone, password_hash, role, location, is_active, created_at)
        VALUES(?,?,?,?,?,?,?,?)
        """,
        (
            "System Administrator",
            email,
            "",
            password_hash,
            "admin",
            "System",
            1,
            now,
        ),
    )


def _ensure_default_officials(cur: sqlite3.Cursor) -> None:
    """Ensure default official accounts exist."""
    officials = [
        ("Sanitation Officer", "sanitation_official@civiccomplaints.gov", "Sanitation@123", "Sanitation"),
        ("Water Supply Officer", "watersupply_official@civiccomplaints.gov", "Water@123", "Water Supply"),
        ("Transportation Officer", "transportation_official@civiccomplaints.gov", "Transportation@123", "Transportation")
    ]
    
    for name, email, password, dept_cat in officials:
        cur.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            continue
            
        cur.execute("SELECT id FROM departments WHERE category = ?", (dept_cat,))
        row = cur.fetchone()
        if not row:
            # Try fuzzy match if exact category fails
            cur.execute("SELECT id FROM departments WHERE name LIKE ?", (f"%{dept_cat}%",))
            row = cur.fetchone()
            if not row:
                continue
        dept_id = row[0]
        
        pwd_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        now = datetime.utcnow().isoformat()
        
        cur.execute("""
            INSERT INTO users(name, email, phone, password_hash, role, location, department_id, is_active, is_approved, created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (name, email, "", pwd_hash, "official", "Head Office", dept_id, 1, 1, now))


def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, definition: str) -> None:
    """Add a column to a table if it does not already exist (idempotent)."""
    cur.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}
    if column not in existing:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def initialize_database() -> None:
    """
    Initialize the database with schema and default data.
    Creates all tables and inserts default admin account and departments.
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        # Read and execute schema
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Execute schema (split by semicolons for SQLite compatibility)
            for statement in schema_sql.split(';'):
                if statement.strip():
                    try:
                        cursor.execute(statement)
                    except sqlite3.OperationalError as e:
                        # Skip if table already exists
                        if 'already exists' not in str(e):
                            raise
            
            # Backfill new authentication/security columns if missing
            _ensure_column(cursor, "users", "failed_login_attempts", "INTEGER DEFAULT 0")
            _ensure_column(cursor, "users", "last_failed_login", "TIMESTAMP")
            _ensure_column(cursor, "users", "captcha_required", "BOOLEAN DEFAULT FALSE")
            _ensure_column(cursor, "sessions", "ip_address", "VARCHAR(45)")
            _ensure_column(cursor, "sessions", "user_agent", "TEXT")

            # Insert default departments
            departments = [
                ('Municipal Corporation - Water Supply', 'Water Supply', None, 'water@municipal.gov', '1800-111-001'),
                ('Municipal Corporation - Sanitation', 'Sanitation', None, 'sanitation@municipal.gov', '1800-111-002'),
                ('Municipal Corporation - Roads & Transport', 'Transportation', None, 'transport@municipal.gov', '1800-111-003')
            ]
            
            for dept in departments:
                cursor.execute("""
                    INSERT OR IGNORE INTO departments (name, category, head_official_id, contact_email, contact_phone)
                    VALUES (?, ?, ?, ?, ?)
                """, dept)
            
            # Create default admin account only if it doesn't exist
            cursor.execute("SELECT id FROM users WHERE email = ?", ('admin@civiccomplaints.gov',))
            if cursor.fetchone() is None:
                admin_password = "Admin@123"
                password_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                
                cursor.execute("""
                    INSERT INTO users (name, email, phone, password_hash, role, location, is_active, is_approved, must_change_password)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'System Administrator',
                    'admin@civiccomplaints.gov',
                    '+919999999999',
                    password_hash,
                    'admin',
                    'Coimbatore',
                    True,
                    True,
                    True  # Force password change on first login
                ))
                logger.info("Default admin account created: admin@civiccomplaints.gov / Admin@123")
            
            logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def init_db(project_root: Optional[Path] = None) -> None:
    """Create all required tables if they do not already exist."""

    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    schema_path = project_root / "config" / "database_schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    schema_sql = schema_path.read_text(encoding="utf-8")

    with get_connection(project_root) as conn:
        cur = conn.cursor()

        # Apply each statement from the schema file (split on semicolons)
        for statement in schema_sql.split(";"):
            stmt = statement.strip()
            if not stmt:
                continue
            try:
                cur.execute(stmt)
            except sqlite3.OperationalError as exc:  # noqa: PERF203
                # Skip benign "already exists" errors so init is idempotent
                if "already exists" not in str(exc):
                    raise

        # Ensure schema updates for dashboards
        _ensure_column(cur, "complaints", "assigned_to", "INTEGER")
        _ensure_column(cur, "departments", "sla_hours", "INTEGER DEFAULT 48")

        # Clustering columns
        _ensure_column(cur, "complaints", "parent_complaint_id", "TEXT")
        _ensure_column(cur, "complaints", "is_clustered", "BOOLEAN DEFAULT FALSE")
        _ensure_column(cur, "complaints", "cluster_size", "INTEGER DEFAULT 1")

        # Seed default departments for routing if they are missing
        departments = [
            (
                "Municipal Corporation - Water Supply",
                "Water Supply",
                None,
                "water@municipal.gov",
                "1800-111-001",
            ),
            (
                "Municipal Corporation - Sanitation",
                "Sanitation",
                None,
                "sanitation@municipal.gov",
                "1800-111-002",
            ),
            (
                "Municipal Corporation - Roads & Transport",
                "Transportation",
                None,
                "transport@municipal.gov",
                "1800-111-003",
            ),
        ]

        cur.executemany(
            """
            INSERT OR IGNORE INTO departments
                (name, category, head_official_id, contact_email, contact_phone)
            VALUES (?, ?, ?, ?, ?)
            """,
            departments,
        )

        # Backfill authentication/security columns for existing DBs
        _ensure_column(cur, "users", "failed_login_attempts", "INTEGER DEFAULT 0")
        _ensure_column(cur, "users", "last_failed_login", "TIMESTAMP")
        _ensure_column(cur, "users", "captcha_required", "BOOLEAN DEFAULT FALSE")
        _ensure_column(cur, "sessions", "ip_address", "VARCHAR(45)")
        _ensure_column(cur, "sessions", "user_agent", "TEXT")

        # Seed a default admin account for first-time setups. In production,
        # the password should be changed immediately after deployment.
        _ensure_default_admin(cur)
        _ensure_default_officials(cur)

        logger.info("Database initialised at %s", _resolve_db_path(project_root))


@dataclass
class User:
    """Simple user representation used in non-auth flows.

    This mirrors a subset of columns from the ``users`` table. The
    authentication subsystem exposes a richer user view including password
    hashes and activation flags.
    """

    id: int
    name: str
    email: str
    phone: str
    role: str
    created_at: str


def get_or_create_user(
    name: str,
    email: str,
    phone: str,
    role: str = "citizen",
    password: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> User:
    """Fetch a user by email or create if it does not exist.
    
    If password is not provided, a dummy hash is used (for auto-created users).
    """
    now = datetime.utcnow().isoformat()
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        if row is not None:
            data = dict(row)
            return User(
                id=int(data["id"]),
                name=str(data.get("name", "")),
                email=str(data.get("email", "")),
                phone=str(data.get("phone", "")),
                role=str(data.get("role", "citizen")),
                created_at=str(data.get("created_at", "")),
            )

        # Generate password hash - use dummy hash for auto-created users
        if password:
            password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        else:
            # Dummy hash for users created without password (e.g., auto-created from complaints)
            password_hash = bcrypt.hashpw(b"dummy_auto_generated", bcrypt.gensalt()).decode()

        cur.execute(
            "INSERT INTO users(name, email, phone, role, password_hash, created_at) VALUES(?,?,?,?,?,?)",
            (name, email, phone, role, password_hash, now),
        )
        user_id = int(cur.lastrowid) if cur.lastrowid is not None else -1
        return User(id=user_id, name=name, email=email, phone=phone, role=role, created_at=now)


def insert_department(
    name: str,
    category: str,
    officials: Optional[List[int]] = None,
    project_root: Optional[Path] = None,
) -> int:
    """Insert a department and return its ID.

    If a department with the same name and category already exists, its
    ID is returned instead of creating a duplicate.
    """
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM departments WHERE name = ? AND category = ?",
            (name, category),
        )
        row = cur.fetchone()
        if row is not None:
            return int(row["id"])

        # Insert department with basic info (officials not stored in schema)
        cur.execute(
            "INSERT INTO departments(name, category) VALUES(?,?)",
            (name, category),
        )
        return int(cur.lastrowid) if cur.lastrowid is not None else -1


def get_department_id_for_category(
    category: str,
    project_root: Optional[Path] = None,
) -> int:
    """Return department ID for a category, using existing or creating default."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        # Try finding ANY department with this category
        cur.execute("SELECT id FROM departments WHERE category = ?", (category,))
        row = cur.fetchone()
        if row:
            return int(row["id"])
            
    default_name = f"Municipal Department - {category}"
    return insert_department(default_name, category, project_root=project_root)


def insert_complaint(
    complaint_id: str,
    user_id: int,
    text: str,
    category: str,
    urgency: str,
    language: str,
    location: str,
    status: str,
    department_id: int,
    parent_complaint_id: Optional[str] = None,
    is_clustered: bool = False,
    cluster_size: int = 1,
    project_root: Optional[Path] = None,
) -> None:
    """Insert a complaint record into the database."""
    now = datetime.utcnow().isoformat()
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO complaints(
                complaint_id, user_id, text, category, urgency, language, location,
                status, department_id, created_at, updated_at,
                parent_complaint_id, is_clustered, cluster_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                complaint_id,
                user_id,
                text,
                category,
                urgency,
                language,
                location,
                status,
                department_id,
                now,
                now,
                parent_complaint_id,
                is_clustered,
                cluster_size,
            ),
        )


def get_complaint(complaint_id: str, project_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Fetch a complaint by ID or return ``None`` if not found."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,))
        row = cur.fetchone()
        if row is None:
            return None

        data = dict(row)
        # Keep the internal row id but expose the external complaint_id via the
        # conventional "id" key for caller compatibility.
        data["row_id"] = data.get("id")
        data["id"] = data.get("complaint_id", data.get("id"))
        return data


def list_open_complaints_for_department(
    department_id: int,
    project_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return all non-resolved complaints for a department."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM complaints
            WHERE department_id = ? AND status != 'Resolved' AND is_clustered = 0
            ORDER BY created_at ASC
            """,
            (department_id,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def list_complaints_by_user(
    user_id: int,
    project_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return all complaints filed by a specific user."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM complaints
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def insert_status_update(
    complaint_id: str,
    status: str,
    remarks: str = "",
    official_id: Optional[int] = None,
    project_root: Optional[Path] = None,
) -> None:
    """Insert a new status update for a complaint and update its row."""
    now = datetime.utcnow().isoformat()
    with get_connection(project_root) as conn:
        cur = conn.cursor()

        # Ensure the legacy DB has the "status" column (older DBs may miss it)
        cur.execute("PRAGMA table_info(status_updates)")
        cols = [r[1] for r in cur.fetchall()]
        if "status" not in cols:
            cur.execute("ALTER TABLE status_updates ADD COLUMN status VARCHAR(50)")

        # Get the integer id for the complaint from the complaint_id string
        cur.execute("SELECT id FROM complaints WHERE complaint_id = ?", (complaint_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Complaint {complaint_id} not found")
        complaint_row_id = int(row[0])

        # First create the status update referencing the internal row id
        # (note: new_status and status are both set for backward compat with schema)
        cur.execute(
            "INSERT INTO status_updates(complaint_id, new_status, status, remarks, updated_by, timestamp) VALUES(?,?,?,?,?,?)",
            (complaint_row_id, status, status, remarks, official_id, now),
        )
        # Then update the complaints table
        cur.execute(
            "UPDATE complaints SET status = ?, updated_at = ? WHERE complaint_id = ?",
            (status, now, complaint_id),
        )


def list_status_updates_for_complaint(
    complaint_id: str,
    project_root: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Return all status updates for a complaint ordered by time."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM complaints WHERE complaint_id = ?", (complaint_id,))
        row = cur.fetchone()
        if row is None:
            return []

        complaint_row_id = int(row[0])
        cur.execute(
            "SELECT * FROM status_updates WHERE complaint_id = ? ORDER BY timestamp ASC",
            (complaint_row_id,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def insert_notification(
    user_id: int,
    message: str,
    complaint_id: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> None:
    """Insert an in-app notification for a user."""
    now = datetime.utcnow().isoformat()
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        complaint_row_id = None
        if complaint_id:
            cur.execute("SELECT id FROM complaints WHERE complaint_id = ?", (complaint_id,))
            row = cur.fetchone()
            if row:
                complaint_row_id = int(row[0])
        cur.execute(
            "INSERT INTO notifications(user_id, complaint_id, message, is_read, created_at) VALUES(?,?,?,?,?)",
            (user_id, complaint_row_id, message, 0, now),
        )


def list_notifications(
    user_id: int,
    include_read: bool = False,
    project_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return notifications for a user."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        if include_read:
            cur.execute("SELECT * FROM notifications WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        else:
            cur.execute(
                "SELECT * FROM notifications WHERE user_id = ? AND is_read = 0 ORDER BY created_at DESC",
                (user_id,),
            )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def mark_notifications_read(user_id: int, project_root: Optional[Path] = None) -> None:
    """Mark all notifications for a user as read."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE notifications SET is_read = 1 WHERE user_id = ?", (user_id,))


def insert_model_prediction(
    complaint_id: str,
    category_prob: Dict[str, float],
    urgency_prob: Dict[str, float],
    shap_summary: Dict[str, Any],
    project_root: Optional[Path] = None,
) -> None:
    """Persist model prediction metadata for auditing and analysis."""
    now = datetime.utcnow().isoformat()
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        # Get the integer id for the complaint from the complaint_id string
        cur.execute("SELECT id FROM complaints WHERE complaint_id = ?", (complaint_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Complaint {complaint_id} not found")
        complaint_row_id = row[0]
        
        # Extract category and urgency predictions
        category_predicted = max(category_prob.keys(), key=lambda k: category_prob[k]) if category_prob else None
        category_confidence = max(category_prob.values()) if category_prob else 0.0
        urgency_predicted = max(urgency_prob.keys(), key=lambda k: urgency_prob[k]) if urgency_prob else None
        urgency_confidence = max(urgency_prob.values()) if urgency_prob else 0.0
        
        cur.execute(
            """
            INSERT INTO model_predictions(
                complaint_id, category_predicted, category_confidence, 
                category_probabilities, urgency_predicted, urgency_confidence, 
                urgency_probabilities, shap_values, timestamp
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                complaint_row_id,
                category_predicted,
                category_confidence,
                json.dumps(category_prob),
                urgency_predicted,
                urgency_confidence,
                json.dumps(urgency_prob),
                json.dumps(shap_summary),
                now,
            ),
        )



def list_active_complaints_by_category(
    category: str,
    project_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return all active (non-resolved) complaints for a category."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM complaints
            WHERE category = ? 
            AND status != 'Resolved'
            AND is_clustered = 0
            ORDER BY created_at DESC
            """,
            (category,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def get_child_complaints(parent_id: str, project_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return all child complaints linked to a parent complaint."""
    with get_connection(project_root) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM complaints
            WHERE parent_complaint_id = ?
            ORDER BY created_at DESC
            """,
            (parent_id,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def execute_query(query: str, params: Tuple = (), fetch: Optional[str] = None) -> Optional[List[Dict[Any, Any]] | Dict[Any, Any]]:
    """
    Execute a SQL query with parameters.
    
    Args:
        query (str): SQL query string
        params (Tuple): Query parameters
        fetch (str): Fetch mode - 'one', 'all', or None for non-SELECT queries
    
    Returns:
        Optional[List[Dict]]: Query results as list of dictionaries, or None
    """
    def _execute_query():
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if fetch == 'one':
                row = cursor.fetchone()
                return dict(row) if row else None
            elif fetch == 'all':
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                return None
    
    # Retry logic for database locks
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return _execute_query()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Database locked, retrying ({attempt + 1}/{max_retries}): {e}")
                time.sleep(1.0 * (attempt + 1))  # Increased exponential backoff
                continue
            else:
                logger.error(f"Query execution error: {e}")
                raise
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise


def insert_record(table: str, data: Dict[str, Any]) -> Optional[int]:
    """
    Insert a record into specified table.
    
    Args:
        table (str): Table name
        data (Dict): Column-value pairs
    
    Returns:
        Optional[int]: ID of inserted record, or None on failure
    """
    def _execute_insert():
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(data.values()))
            return cursor.lastrowid
    
    # Retry logic for database locks
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = _execute_insert()
            return result
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Database locked, retrying ({attempt + 1}/{max_retries}): {e}")
                time.sleep(1.0 * (attempt + 1))  # Increased exponential backoff
                continue
            else:
                logger.error(f"Insert error in {table}: {e}")
                raise
        except Exception as e:
            logger.error(f"Insert error in {table}: {e}")
            raise


def update_record(table: str, record_id: Any, data: Dict[str, Any], id_field: str = "id") -> bool:
    """
    Update a record in specified table.
    
    Args:
        table (str): Table name
        record_id (Any): Record ID to update
        data (Dict): Column-value pairs to update
        id_field (str, optional): Name of the ID column. Defaults to "id".
    
    Returns:
        bool: True if successful, False otherwise
    """
    def _execute_update():
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {id_field} = ?"
        
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(data.values()) + (record_id,))
            return cursor.rowcount > 0
    
    # Retry logic for database locks
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = _execute_update()
            return bool(result)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Database locked, retrying ({attempt + 1}/{max_retries}): {e}")
                time.sleep(1.0 * (attempt + 1))  # Increased exponential backoff
                continue
            else:
                logger.error(f"Update error in {table}: {e}")
                return False
        except Exception as e:
            logger.error(f"Update error in {table}: {e}")
            return False
    return False


def cleanup_expired_sessions() -> None:
    """Remove expired sessions from database."""
    try:
        query = "DELETE FROM sessions WHERE expires_at < ? OR is_active = FALSE"
        execute_query(query, (datetime.now(),))
        logger.info("Expired sessions cleaned up")
    except Exception as e:
        logger.error(f"Session cleanup error: {e}")


def update_complaint_assignment(complaint_id: str, official_id: int) -> bool:
    """Assign a complaint to a specific official."""
    try:
        # Update assignment
        success = update_record("complaints", complaint_id, {"assigned_to": official_id, "status": "Assigned"}, id_field="complaint_id")
        if not success:
            logger.error(f"Failed to update complaint {complaint_id} assignment.")
            return False
            
        # Log status change using the helper that handles internal IDs
        insert_status_update(
            complaint_id=complaint_id,
            status="Assigned",
            remarks=f"Assigned to official ID {official_id}",
            official_id=official_id
        )
        return True
    except Exception as e:
        logger.error(f"Failed to assign complaint: {e}")
        return False


def get_all_complaints_global(project_root: Optional[str] = None) -> List[Dict[Any, Any]]:
    """Retrieve all complaints for Admin view."""
    result = execute_query("""
        SELECT c.*, d.name as department_name, u.name as official_name, citizen.name as name
        FROM complaints c
        LEFT JOIN departments d ON c.department_id = d.id
        LEFT JOIN users u ON c.assigned_to = u.id
        LEFT JOIN users citizen ON c.user_id = citizen.id
        ORDER BY c.created_at DESC
    """, fetch="all")
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return []


def get_users_by_role(role: Optional[str] = None) -> List[Dict[Any, Any]]:
    """Retrieve users, optionally filtered by role."""
    sql = """
        SELECT u.id, u.name, u.email, u.phone, u.role, u.department_id, u.is_active, u.is_approved, u.last_login, d.name as department_name
        FROM users u
        LEFT JOIN departments d ON u.department_id = d.id
    """
    params = []
    if role:
        sql += " WHERE u.role = ?"
        params.append(role)
    sql += " ORDER BY u.created_at DESC"
    result = execute_query(sql, tuple(params), fetch="all")
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return []


def update_user_status(user_id: int, is_active: bool, is_approved: bool) -> bool:
    """Update user activation and approval status."""
    try:
        update_record("users", user_id, {"is_active": is_active, "is_approved": is_approved})
        return True
    except Exception as e:
        logger.error(f"Failed to update user status: {e}")
        return False


__all__ = [
    "init_db",
    "get_or_create_user",
    "get_department_id_for_category",
    "insert_complaint",
    "get_complaint",
    "list_open_complaints_for_department",
    "insert_status_update",
    "list_status_updates_for_complaint",
    "insert_notification",
    "list_notifications",
    "mark_notifications_read",
    "insert_model_prediction",
    "update_complaint_assignment",
    "get_all_complaints_global",
    "get_users_by_role",
    "update_user_status",
    "list_complaints_by_user",
]
