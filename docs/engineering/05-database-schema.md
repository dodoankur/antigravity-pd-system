# 05. Database Schema

## Tables

### `measurements`
- `id`: UUID (Primary Key)
- `session_id`: VARCHAR (Index)
- `pd_binocular`: FLOAT
- `pd_left`: FLOAT
- `pd_right`: FLOAT
- `confidence_score`: FLOAT
- `metadata`: JSONB (Device info, browser version, lighting score)
- `created_at`: TIMESTAMP

### `sessions`
- `id`: UUID
- `user_id`: UUID (Optional)
- `status`: ENUM ('active', 'completed', 'failed')
- `error_log`: TEXT
