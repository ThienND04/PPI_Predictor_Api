from pydantic import BaseModel, EmailStr, Field, constr


class RegisterInput(BaseModel):
    email: EmailStr
    full_name: constr(strip_whitespace=True, min_length=1) = Field(...)
    password: constr(min_length=6)


class VerifyInput(BaseModel):
    email: EmailStr
    code: constr(strip_whitespace=True, min_length=1, max_length=12)


class ResendCodeInput(BaseModel):
    email: EmailStr


class LoginInput(BaseModel):
    email: EmailStr
    password: constr(min_length=1)


class ForgotPasswordInput(BaseModel):
    email: EmailStr


class ResetPasswordInput(BaseModel):
    email: EmailStr
    otp_code: constr(min_length=6, max_length=6)
    new_password: constr(min_length=6)


class ChangePasswordInput(BaseModel):
    email: EmailStr
    old_password: constr(min_length=1)
    new_password: constr(min_length=6)


