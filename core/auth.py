import jwt
import datetime
from config import JWT_ALGORITHM, JWT_EXPIRATION_HOURS, CLASSIFICATION_ACCESS, ROLES
from core.database import get_db_connection, verify_password

def authenticate_user(username, password):
    """Authenticate a user against the DB and return a signed JWT token if valid."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username, password_hash, role, department, jwt_secret FROM users WHERE username = ?;", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None
        
    stored_hash = user["password_hash"]
    if not verify_password(stored_hash, password):
        return None
        
    # Generate token payload
    payload = {
        "sub": user["username"],
        "role": user["role"],
        "dept": user["department"],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    
    # Sign JWT with the user's unique database secret key
    token = jwt.encode(payload, user["jwt_secret"], algorithm=JWT_ALGORITHM)
    return token

def verify_token(token, username):
    """Verify a signed JWT token for a specific user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT jwt_secret, role, department FROM users WHERE username = ?;", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None
        
    try:
        payload = jwt.decode(token, user["jwt_secret"], algorithms=[JWT_ALGORITHM])
        if payload["sub"] != username:
            return None
        return {
            "username": payload["sub"],
            "role": payload["role"],
            "department": payload["dept"]
        }
    except jwt.ExpiredSignatureError:
        print("Token has expired.")
        return None
    except jwt.InvalidTokenError:
        print("Invalid token.")
        return None

def check_classification_access(user_role: str, classification: str) -> bool:
    """Enforce strict department/role classification permissions."""
    if user_role not in CLASSIFICATION_ACCESS:
        return False
    return classification in CLASSIFICATION_ACCESS[user_role]

def get_allowed_classifications(user_role: str) -> list:
    """Get the full list of allowed data classifications for a role."""
    return CLASSIFICATION_ACCESS.get(user_role, ["Public"])
