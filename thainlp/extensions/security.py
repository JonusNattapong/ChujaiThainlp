from typing import Optional, List, Dict
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pydantic import BaseModel
from cryptography.fernet import Fernet
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from enum import Enum

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class SecurityConfig:
    SECRET_KEY: str = "your-secret-key"  # ควรย้ายไปอยู่ใน environment variables
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ENCRYPTION_KEY: bytes = Fernet.generate_key()  # สำหรับ end-to-end encryption

class Token(BaseModel):
    access_token: str
    token_type: str
    role: Role

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[Role] = None

class User(BaseModel):
    username: str
    role: Role
    disabled: Optional[bool] = None

class RBACPolicy:
    _policies: Dict[Role, List[str]] = {
        Role.ADMIN: ["read", "write", "delete", "manage"],
        Role.USER: ["read", "write"],
        Role.GUEST: ["read"]
    }

    @classmethod
    def has_permission(cls, role: Role, action: str) -> bool:
        return action in cls._policies.get(role, [])

class SecurityManager:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.fernet = Fernet(SecurityConfig.ENCRYPTION_KEY)
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def create_access_token(self, data: dict, role: Role,
                          expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
            
        to_encode.update({
            "exp": expire,
            "role": role.value
        })
        return jwt.encode(
            to_encode,
            SecurityConfig.SECRET_KEY,
            algorithm=SecurityConfig.ALGORITHM
        )

    def verify_token(self, token: str) -> Optional[TokenData]:
        try:
            payload = jwt.decode(
                token,
                SecurityConfig.SECRET_KEY,
                algorithms=[SecurityConfig.ALGORITHM]
            )
            username: str = payload.get("sub")
            role: str = payload.get("role", Role.GUEST.value)
            
            if username is None:
                return None
                
            return TokenData(username=username, role=Role(role))
        except JWTError:
            return None

    def encrypt_data(self, data: str) -> str:
        """End-to-end encryption สำหรับข้อมูลที่ sensitive"""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """ถอดรหัสข้อมูลที่เข้ารหัสไว้"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

class AuditLog(BaseModel):
    timestamp: datetime
    user: str
    action: str
    resource: str
    status: str
    details: Optional[Dict] = None

class SecurityAuditor:
    def __init__(self):
        self.logs: List[AuditLog] = []
        
    def log_action(self, user: str, action: str, resource: str,
                   status: str, details: Optional[Dict] = None):
        """บันทึก audit log สำหรับการตรวจสอบความปลอดภัย"""
        log = AuditLog(
            timestamp=datetime.utcnow(),
            user=user,
            action=action,
            resource=resource,
            status=status,
            details=details
        )
        self.logs.append(log)
        
    def get_user_actions(self, user: str) -> List[AuditLog]:
        """ดึงประวัติการกระทำของผู้ใช้"""
        return [log for log in self.logs if log.user == user]

    def get_resource_access(self, resource: str) -> List[AuditLog]:
        """ดึงประวัติการเข้าถึงทรัพยากร"""
        return [log for log in self.logs if log.resource == resource] 