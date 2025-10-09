import os
import smtplib
from email.message import EmailMessage


def send_verification_email(to_email: str, code: str, optional_link: str = None) -> None:
    smtp_host = os.getenv('SMTP_HOST') or os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    email_from = os.getenv('EMAIL_FROM', smtp_user or '')
    exp_min = os.getenv('VERIFICATION_EXP_MIN', '10')

    if not (smtp_host and smtp_port and smtp_user and smtp_password and email_from):
        raise RuntimeError('SMTP configuration is incomplete')

    msg = EmailMessage()
    msg['Subject'] = 'Xác minh email - PPI Predictor'
    msg['From'] = email_from
    msg['To'] = to_email

    body_lines = [
        'Xin chào,',
        f'Mã xác minh của bạn là: {code}',
        f'Mã có hiệu lực trong {exp_min} phút.',
    ]
    if optional_link:
        body_lines.append(f'Bạn cũng có thể xác minh bằng liên kết: {optional_link}')
    body_lines.append('Nếu bạn không yêu cầu đăng ký, vui lòng bỏ qua email này.')
    msg.set_content('\n'.join(body_lines))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def send_reset_otp_email(to_email: str, otp_code: str, exp_min: int) -> None:
    smtp_host = os.getenv('SMTP_HOST') or os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    email_from = os.getenv('EMAIL_FROM', smtp_user or '')

    if not (smtp_host and smtp_port and smtp_user and smtp_password and email_from):
        raise RuntimeError('SMTP configuration is incomplete')

    msg = EmailMessage()
    msg['Subject'] = 'Password Reset OTP - PPI Predictor'
    msg['From'] = email_from
    msg['To'] = to_email

    body_lines = [
        'Hello,',
        f'Your password reset OTP is: {otp_code}',
        f'This code is valid for {exp_min} minutes.',
        'If you did not request a password reset, please ignore this email.'
    ]
    msg.set_content('\n'.join(body_lines))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)

