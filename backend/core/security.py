from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
import re
import hashlib
import secrets
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Security constants
MIN_PASSWORD_LENGTH = 8
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against its hash.
    
    Args:
        plain_password: The plain text password
        hashed_password: The hashed password to verify against
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: The plain text password to hash
        
    Returns:
        The hashed password
    """
    return pwd_context.hash(password)

def validate_password_strength(password: str) -> Dict[str, Any]:
    """
    Validate password strength and return detailed feedback.
    
    Args:
        password: The password to validate
        
    Returns:
        Dictionary with validation results and feedback
    """
    result = {
        "is_valid": True,
        "score": 0,
        "feedback": [],
        "requirements_met": {
            "length": False,
            "uppercase": False,
            "lowercase": False,
            "digits": False,
            "special_chars": False
        }
    }
    
    # Check length
    if len(password) >= MIN_PASSWORD_LENGTH:
        result["requirements_met"]["length"] = True
        result["score"] += 1
    else:
        result["feedback"].append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")
    
    # Check for uppercase letters
    if re.search(r'[A-Z]', password):
        result["requirements_met"]["uppercase"] = True
        result["score"] += 1
    else:
        result["feedback"].append("Password must contain at least one uppercase letter")
    
    # Check for lowercase letters
    if re.search(r'[a-z]', password):
        result["requirements_met"]["lowercase"] = True
        result["score"] += 1
    else:
        result["feedback"].append("Password must contain at least one lowercase letter")
    
    # Check for digits
    if re.search(r'\d', password):
        result["requirements_met"]["digits"] = True
        result["score"] += 1
    else:
        result["feedback"].append("Password must contain at least one digit")
    
    # Check for special characters
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        result["requirements_met"]["special_chars"] = True
        result["score"] += 1
    else:
        result["feedback"].append("Password must contain at least one special character")
    
    # Overall validation
    result["is_valid"] = all(result["requirements_met"].values())
    
    return result

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary containing user data to encode
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        data: Dictionary containing user data to encode
        
    Returns:
        Encoded JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT access token.
    
    Args:
        token: The JWT token to decode
        
    Returns:
        Decoded payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check token type
        if payload.get("type") != "access":
            logger.warning("Invalid token type")
            return None
            
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None

def decode_refresh_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT refresh token.
    
    Args:
        token: The JWT refresh token to decode
        
    Returns:
        Decoded payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check token type
        if payload.get("type") != "refresh":
            logger.warning("Invalid refresh token type")
            return None
            
        return payload
    except JWTError as e:
        logger.warning(f"JWT refresh token decode error: {e}")
        return None

def sanitize_input(input_data: str) -> str:
    """
    Sanitize input data to prevent injection attacks.
    
    Args:
        input_data: The input string to sanitize
        
    Returns:
        Sanitized input string
    """
    if not isinstance(input_data, str):
        return str(input_data)
    
    # Remove null bytes and control characters
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', input_data)
    
    # HTML escape
    import html
    sanitized = html.escape(sanitized.strip())
    
    # Remove potential SQL injection patterns
    sql_patterns = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)',
        r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
        r'(\b(OR|AND)\s+[\'"]?\w+[\'"]?\s*=\s*[\'"]?\w+[\'"]?)',
    ]
    
    for pattern in sql_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()

def generate_csrf_token() -> str:
    """
    Generate a CSRF token.
    
    Returns:
        A random CSRF token
    """
    return secrets.token_urlsafe(32)

def verify_csrf_token(token: str, stored_token: str) -> bool:
    """
    Verify a CSRF token.
    
    Args:
        token: The token to verify
        stored_token: The stored token to compare against
        
    Returns:
        True if tokens match, False otherwise
    """
    return secrets.compare_digest(token, stored_token)

def hash_sensitive_data(data: str) -> str:
    """
    Hash sensitive data for logging or storage.
    
    Args:
        data: The sensitive data to hash
        
    Returns:
        SHA-256 hash of the data
    """
    return hashlib.sha256(data.encode()).hexdigest()

def is_safe_filename(filename: str) -> bool:
    """
    Check if a filename is safe (no path traversal attempts).
    
    Args:
        filename: The filename to check
        
    Returns:
        True if filename is safe, False otherwise
    """
    # Check for path traversal attempts
    dangerous_patterns = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    
    for pattern in dangerous_patterns:
        if pattern in filename:
            return False
    
    # Check length
    if len(filename) > 255:
        return False
    
    return True

def rate_limit_check(identifier: str, attempts: int, window_minutes: int = 15) -> bool:
    """
    Check if an identifier has exceeded rate limits.
    
    Args:
        identifier: Unique identifier (IP, user ID, etc.)
        attempts: Current number of attempts
        window_minutes: Time window in minutes
        
    Returns:
        True if within limits, False if rate limited
    """
    # This is a simple implementation
    # In production, you'd use Redis or similar for distributed rate limiting
    return attempts < MAX_LOGIN_ATTEMPTS