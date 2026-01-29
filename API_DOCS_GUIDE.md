# API Documentation - Hướng dẫn sử dụng

## 🚀 Cách truy cập API Documentation

Sau khi chạy Docker container, truy cập Swagger UI tại:

**URL:** [http://localhost:5000/apidocs/](http://localhost:5000/apidocs/)

## 📋 Các bước sử dụng

### 1. Khởi động lại Docker để cài đặt dependencies mới

```bash
# Rebuild Docker image với flasgger mới
docker-compose down
docker-compose build
docker-compose up -d

# Hoặc nếu dùng Docker trực tiếp
docker build -t ppi-predictor-api .
docker run -p 5000:5000 ppi-predictor-api
```

### 2. Truy cập Swagger UI

Mở browser và vào: `http://localhost:5000/apidocs/`

Bạn sẽ thấy giao diện Swagger UI với đầy đủ documentation của tất cả API endpoints.

### 3. Test API trực tiếp trên Swagger UI

#### Bước 1: Test endpoint không cần authentication
- Chọn endpoint `/auth/register` hoặc `/auth/login`
- Click **"Try it out"**
- Điền thông tin vào request body
- Click **"Execute"**
- Xem response bên dưới

#### Bước 2: Lấy JWT token
- Test endpoint `/auth/login` với credentials hợp lệ
- Copy token từ response (ví dụ: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`)

#### Bước 3: Sử dụng authentication cho endpoints khác
- Scroll lên đầu trang, click nút **"Authorize"** (biểu tượng ổ khóa)
- Trong popup, nhập: `Bearer <your_token>` (thay `<your_token>` bằng token vừa copy)
- Click **"Authorize"**
- Click **"Close"**

#### Bước 4: Test các endpoint có authentication
- Các endpoint như `/predict`, `/history` giờ sẽ tự động gửi kèm token
- Chọn endpoint, click **"Try it out"**, điền parameters
- Click **"Execute"** để xem kết quả

## 📚 Cấu trúc Documentation

### Authentication Endpoints (`/api/auth`)
- `POST /auth/register` - Đăng ký tài khoản
- `POST /auth/login` - Đăng nhập (nhận JWT token)
- `POST /auth/verify` - Xác minh email
- `POST /auth/resend-code` - Gửi lại mã xác minh
- `POST /auth/forgot-password` - Yêu cầu reset mật khẩu
- `POST /auth/reset-password` - Reset mật khẩu với OTP
- `POST /auth/change-password` - Đổi mật khẩu

### Prediction Endpoints (`/api/predict`)
- `POST /predict` - Dự đoán PPI đơn lẻ (Rate limit: 10/phút)
- `POST /predict/batch` - Dự đoán hàng loạt từ file (Rate limit: 3/phút)

### History Endpoints (`/api/history`)
- `GET /history` - Lấy tất cả lịch sử dự đoán
- `GET /history/{id}` - Lấy 1 bản ghi cụ thể
- `DELETE /history/{id}` - Xóa 1 bản ghi
- `DELETE /history` - Xóa toàn bộ lịch sử

## 🔐 Authentication Flow

```
1. POST /auth/register → Đăng ký user mới
2. POST /auth/login → Nhận JWT token
3. Sử dụng token trong header: Authorization: Bearer <token>
4. Gọi các API khác với token này
```

## 📝 Ví dụ Request/Response

### Login Example
```json
// Request
POST /api/auth/login
{
  "email": "user@example.com",
  "password": "password123"
}

// Response
{
  "message": "Login successful",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Predict Example
```json
// Request (với Authorization header)
POST /api/predict
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
{
  "id1": "P12345",
  "seq1": "MKTAYIAKQRQISFVK...",
  "id2": "Q98765",
  "seq2": "MAVRSYKDRVKVVLD...",
  "model": "MCAPST5"
}

// Response
{
  "protein1": {"id": "P12345"},
  "protein2": {"id": "Q98765"},
  "model": "MCAPST5",
  "score": 0.7845,
  "label": "interaction",
  "threshold": 0.5,
  "timestamp": "2026-01-29T10:30:00Z"
}
```

## 🎯 Features của Swagger UI

- ✅ **Interactive Testing** - Test API trực tiếp trên browser
- ✅ **Request/Response Examples** - Xem ví dụ cho từng endpoint
- ✅ **Schema Validation** - Xem chi tiết về request/response schemas
- ✅ **Authentication Support** - Dễ dàng test với JWT tokens
- ✅ **Rate Limiting Info** - Hiển thị thông tin về rate limits
- ✅ **Error Codes** - Liệt kê tất cả các error codes có thể xảy ra

## 🔧 Troubleshooting

### Không thấy Swagger UI?
- Kiểm tra Docker container đang chạy: `docker ps`
- Kiểm tra logs: `docker-compose logs -f`
- Đảm bảo port 5000 không bị conflicts

### 401 Unauthorized Error?
- Kiểm tra JWT token còn hiệu lực (expires sau 60 giờ)
- Đảm bảo đã click nút "Authorize" và nhập đúng format: `Bearer <token>`
- Login lại để lấy token mới

### API không response?
- Kiểm tra database connection trong `.env`
- Xem logs của Docker container
- Đảm bảo rate limit không bị vượt quá

## 📄 Export OpenAPI Spec

Để export OpenAPI specification (JSON format):

**URL:** [http://localhost:5000/apispec.json](http://localhost:5000/apispec.json)

File này có thể import vào:
- Postman
- Insomnia
- API clients khác

## 🌐 CORS Configuration

Swagger UI được cấu hình CORS để có thể truy cập từ:
- `http://localhost:5000/apidocs/*` - Swagger UI
- `http://localhost:5173` - Frontend development server
- `http://127.0.0.1:5173` - Frontend development server

## 💡 Tips

1. **Organize by Tags** - Endpoints được nhóm theo Authentication, Prediction, History
2. **Try Examples** - Mỗi endpoint có sẵn example data để test nhanh
3. **Check Rate Limits** - Chú ý rate limits để tránh bị 429 errors
4. **Save Token** - Swagger UI sẽ nhớ token trong session hiện tại
5. **View Raw JSON** - Click "Model" tab để xem raw JSON schema

---

**Chúc bạn sử dụng API documentation hiệu quả! 🎉**
