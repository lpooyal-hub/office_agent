import json
import os
import secrets
from dataclasses import dataclass
from typing import Mapping

from fastapi import HTTPException

ROLES = ("viewer", "editor", "admin")
ROLE_LEVELS = {role: index for index, role in enumerate(ROLES)}


@dataclass(frozen=True)
class AuthenticatedUser:
    role: str
    token_name: str = ""

    @property
    def owner(self):
        return self.token_name or self.role


def _load_role_codes_from_json(raw_value):
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ACCESS_CODES는 JSON 객체 형식이어야 합니다.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("ACCESS_CODES는 역할별 코드를 담은 JSON 객체여야 합니다.")
    return {str(role).lower(): str(code) for role, code in parsed.items() if code}


def _load_role_codes_from_pairs(raw_value):
    role_codes = {}
    if not raw_value:
        return role_codes
    for item in raw_value.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise RuntimeError("ROLE_ACCESS_CODES는 role:code 쌍을 쉼표로 구분해야 합니다.")
        role, code = item.split(":", 1)
        role_codes[role.strip().lower()] = code.strip()
    return role_codes


def load_role_codes(environ: Mapping[str, str] | None = None):
    environ = environ or os.environ
    role_codes = {}
    role_codes.update(_load_role_codes_from_json(environ.get("ACCESS_CODES", "")))
    role_codes.update(_load_role_codes_from_pairs(environ.get("ROLE_ACCESS_CODES", "")))

    for role in ROLES:
        value = environ.get(f"ACCESS_CODE_{role.upper()}", "")
        if value:
            role_codes[role] = value

    legacy_code = environ.get("ACCESS_CODE", "")
    if legacy_code and not role_codes:
        role_codes["admin"] = legacy_code

    invalid_roles = sorted(set(role_codes) - set(ROLES))
    if invalid_roles:
        raise RuntimeError(f"지원하지 않는 역할 코드입니다: {', '.join(invalid_roles)}")
    return role_codes


def authenticate_access_code(access_code, role_codes):
    if not role_codes:
        raise HTTPException(status_code=503, detail="서비스 접속 코드가 설정되어 있지 않습니다.")

    candidate = access_code or ""
    for role, expected_code in role_codes.items():
        if secrets.compare_digest(candidate, expected_code):
            return AuthenticatedUser(role=role, token_name=role)

    raise HTTPException(status_code=403, detail="접속 코드가 올바르지 않습니다.")


def require_role(user: AuthenticatedUser, minimum_role: str):
    if minimum_role not in ROLE_LEVELS:
        raise ValueError(f"알 수 없는 역할입니다: {minimum_role}")
    if ROLE_LEVELS[user.role] < ROLE_LEVELS[minimum_role]:
        raise HTTPException(status_code=403, detail=f"{minimum_role} 이상의 권한이 필요합니다.")
    return user


def roles_at_least(minimum_role):
    if minimum_role not in ROLE_LEVELS:
        raise ValueError(f"알 수 없는 역할입니다: {minimum_role}")
    return [role for role in ROLES if ROLE_LEVELS[role] >= ROLE_LEVELS[minimum_role]]


def normalize_allowed_roles(allowed_roles=None, minimum_role="viewer"):
    roles = allowed_roles or roles_at_least(minimum_role)
    normalized = []
    for role in roles:
        role = str(role).strip().lower()
        if role not in ROLE_LEVELS:
            raise ValueError(f"지원하지 않는 역할입니다: {role}")
        if role not in normalized:
            normalized.append(role)
    return normalized


def build_document_where_filter(user: AuthenticatedUser):
    role_flag = f"allowed_role_{user.role}"
    return {"$or": [{"visibility": "public"}, {role_flag: True}, {"owner": user.owner}]}
